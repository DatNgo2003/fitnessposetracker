import asyncio
import json
import logging
import time
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect as ws_connect

from app.core.config import get_settings
from app.core.const import (
    SONIOX_WEBSOCKET_URL,
    SONIOX_MODEL,
    SONIOX_AUDIO_FORMAT,
    SONIOX_LANGUAGE_HINTS,
)
from app.modules.llm.gemini import gemini_manager
from app.modules.llm.tts_google import google_tts_manager

_logger = logging.getLogger(__name__)


class SonioxSTTManager:
    """Manager for Soniox Speech-to-Text WebSocket connections"""

    def __init__(self):
        self.settings = get_settings()
        self.llm_debounce_delay = (
            2.0  # Wait 2 seconds after last final token before triggering LLM
        )
        self.llm_timers = {}  # Track debounce timers per client
        self.processing_clients = {}  # Track clients currently processing LLM/TTS response

    def _get_soniox_config(self) -> dict:
        """Get Soniox configuration"""
        return {
            "api_key": self.settings.soniox_api_key,
            "model": SONIOX_MODEL,
            "audio_format": SONIOX_AUDIO_FORMAT,
            "sample_rate": self.settings.stt_sample_rate,
            "num_channels": self.settings.stt_channels,
            "language_hints": SONIOX_LANGUAGE_HINTS,
            "enable_endpoint_detection": True,
        }

    async def _process_llm_response(
        self, transcript: str, client_ws: WebSocket, client_id: str
    ) -> None:
        """
        Process transcript through LLM and send response to client

        Args:
            transcript: Final transcript from STT
            client_ws: WebSocket connection to client
            client_id: Unique client identifier
        """
        try:
            # Skip empty or very short transcripts
            if not transcript or len(transcript.strip()) < 3:
                return

            # Mark client as processing (turn-taking: ignore new audio during processing)
            self.processing_clients[client_id] = True
            _logger.info(f"🔒 Turn-taking: Client {client_id} now processing, ignoring new audio")

            # Send processing_started signal to frontend
            await client_ws.send_json({
                "type": "processing_started",
                "message": "Đang xử lý câu trả lời..."
            })

            _logger.info(f"Processing LLM for transcript: {transcript}")
            start_time = time.time()

            # Generate response using Gemini
            ai_response = await gemini_manager.generate_response(transcript)

            processing_time = time.time() - start_time

            if ai_response:
                # Send LLM response to client
                llm_response = {
                    "type": "llm_response",
                    "user_text": transcript,
                    "ai_response": ai_response,
                    "processing_time": processing_time,
                }

                await client_ws.send_json(llm_response)
                _logger.info(
                    f"LLM response sent in {processing_time:.2f}s: {ai_response}"
                )

                # Automatically generate TTS audio (pass client_id for turn-taking)
                asyncio.create_task(self._process_tts_response(ai_response, client_ws, client_id))
            else:
                _logger.warning("No response generated from LLM")
                # Release lock if no TTS will be generated
                self._release_processing_lock(client_id, client_ws)

        except Exception as e:
            _logger.error(f"Error processing LLM response: {e}")
            # Release lock on error
            self._release_processing_lock(client_id, client_ws)
            # Send error to client
            await client_ws.send_json(
                {
                    "type": "llm_error",
                    "error_message": f"LLM processing failed: {str(e)}",
                }
            )

    def _release_processing_lock(self, client_id: str, client_ws: WebSocket) -> None:
        """Release the processing lock for a client and send processing_complete signal"""
        if client_id in self.processing_clients:
            del self.processing_clients[client_id]
            _logger.info(f"🔓 Turn-taking: Client {client_id} processing complete, ready for new input")
            # Send processing_complete signal asynchronously
            asyncio.create_task(self._send_processing_complete(client_ws))

    async def _send_processing_complete(self, client_ws: WebSocket) -> None:
        """Send processing complete signal to client"""
        try:
            await client_ws.send_json({
                "type": "processing_complete",
                "message": "Sẵn sàng nhận câu hỏi tiếp theo"
            })
        except Exception as e:
            _logger.error(f"Error sending processing_complete: {e}")

    def is_client_processing(self, client_id: str) -> bool:
        """Check if a client is currently processing a response"""
        return self.processing_clients.get(client_id, False)

    async def _process_tts_response(self, text: str, client_ws: WebSocket, client_id: str) -> None:
        """
        Process text through TTS and send audio URL to client

        Args:
            text: Text to convert to speech
            client_ws: WebSocket connection to client
            client_id: Unique client identifier for turn-taking
        """
        try:
            _logger.info(f"🔊 Processing TTS for text: {text[:50]}...")

            # Send immediate response to client that TTS is processing
            await client_ws.send_json(
                {"type": "tts_processing", "message": "Đang tạo âm thanh phản hồi..."}
            )

            # Process TTS in background without blocking (pass client_id for turn-taking)
            asyncio.create_task(self._background_tts_processing(text, client_ws, client_id))

        except Exception as e:
            _logger.error(f"Error starting TTS processing: {e}")
            await client_ws.send_json(
                {
                    "type": "tts_error",
                    "error_message": f"TTS processing failed: {str(e)}",
                }
            )

    async def _background_tts_processing(self, text: str, client_ws: WebSocket, client_id: str) -> None:
        """Background TTS processing that doesn't block the main flow"""
        try:
            start_time = time.time()

            # Call TTS (using Google Cloud TTS now)
            audio_filename = await google_tts_manager.text_to_speech(text)

            processing_time = time.time() - start_time

            if audio_filename:
                audio_url = f"/audio/{audio_filename}"

                tts_response = {
                    "type": "tts_response",
                    "text": text,
                    "audio_url": audio_url,
                    "audio_filename": audio_filename,
                    "processing_time": processing_time,
                }

                await client_ws.send_json(tts_response)
                _logger.info(
                    f"TTS response sent in {processing_time:.2f}s: {audio_filename}"
                )
            else:
                _logger.warning("No audio generated from TTS")
                # Send notification to client that TTS failed but conversation continues
                await client_ws.send_json(
                    {
                        "type": "tts_info",
                        "message": "Không thể tạo âm thanh lúc này, nhưng cuộc trò chuyện vẫn tiếp tục bình thường.",
                    }
                )

            # Release processing lock after TTS is done (turn-taking complete)
            self._release_processing_lock(client_id, client_ws)

        except Exception as e:
            _logger.error(f"Error in background TTS processing: {e}")
            # Release processing lock on error
            self._release_processing_lock(client_id, client_ws)
            # Send error to client
            await client_ws.send_json(
                {
                    "type": "tts_error",
                    "error_message": f"TTS processing failed: {str(e)}",
                }
            )

    def _schedule_llm_processing(
        self,
        transcript: str,
        client_ws: WebSocket,
        client_id: str,
        final_tokens_ref: list,
    ) -> None:
        """
        Schedule LLM processing with debounce to avoid triggering on every final token

        Args:
            transcript: Current transcript
            client_ws: WebSocket connection
            client_id: Unique client identifier
        """
        # Cancel existing timer if any
        if client_id in self.llm_timers:
            self.llm_timers[client_id].cancel()

        # Schedule new LLM processing after delay
        async def delayed_llm_processing():
            try:
                await asyncio.sleep(self.llm_debounce_delay)
                # Remove timer from tracking
                if client_id in self.llm_timers:
                    del self.llm_timers[client_id]
                # Process LLM (pass client_id for turn-taking)
                await self._process_llm_response(transcript, client_ws, client_id)
                # Reset final_tokens after processing to start fresh for next conversation
                final_tokens_ref.clear()
                _logger.info("Reset transcript buffer for next conversation")
            except asyncio.CancelledError:
                _logger.debug(f"LLM processing cancelled for client {client_id}")
            except Exception as e:
                _logger.error(f"Error in delayed LLM processing: {e}")

        # Create and store timer
        timer = asyncio.create_task(delayed_llm_processing())
        self.llm_timers[client_id] = timer

        _logger.info(
            f"Scheduled LLM processing in {self.llm_debounce_delay}s for: {transcript[:50]}..."
        )

    async def handle_websocket(self, client_ws: WebSocket) -> None:
        """
        Handle WebSocket connection from frontend client.
        Receives audio chunks from FE and sends back transcription results.
        """
        await client_ws.accept()
        _logger.info("STT WebSocket client connected")

        final_tokens = []
        soniox_ws: Optional[object] = None
        client_id = f"client_{id(client_ws)}"  # Unique client identifier

        try:
            # Connect to Soniox
            _logger.info("Connecting to Soniox STT service...")
            async with ws_connect(SONIOX_WEBSOCKET_URL) as soniox_ws:
                # Send configuration
                config = self._get_soniox_config()
                await soniox_ws.send(json.dumps(config))
                _logger.info("Connected to Soniox, waiting for audio...")

                silence_start: Optional[float] = None
                session_closing = False  # Flag to track when client requests close

                # Start tasks for bidirectional communication
                async def receive_from_client():
                    """Receive audio from client and send to Soniox"""
                    nonlocal silence_start
                    try:
                        while True:
                            # Receive audio data from client
                            message = await client_ws.receive()

                            if message["type"] == "websocket.disconnect":
                                _logger.info("Client disconnected")
                                break

                            # Handle text messages (control signals)
                            if "text" in message:
                                data = json.loads(message["text"])
                                if data.get("type") == "close":
                                    nonlocal session_closing
                                    session_closing = True
                                    _logger.info("Client requested close")
                                    await soniox_ws.send(b"")
                                    break
                                continue

                            # Handle binary audio data
                            if "bytes" in message:
                                # Turn-taking: Ignore audio during processing phase
                                if self.is_client_processing(client_id):
                                    # Still receive but don't process - keeps WebSocket alive
                                    continue

                                audio_bytes = message["bytes"]

                                # Convert to numpy array for RMS calculation
                                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                                rms = np.abs(audio_np).mean()

                                # Silent detection
                                if rms < self.settings.stt_silence_threshold:
                                    if silence_start is None:
                                        silence_start = time.time()
                                    elif (
                                        time.time() - silence_start
                                        > self.settings.stt_long_silence
                                    ):
                                        _logger.info(
                                            "Long silence detected, closing session"
                                        )
                                        await client_ws.send_json(
                                            {
                                                "type": "info",
                                                "message": "Session closed due to long silence",
                                            }
                                        )
                                        await soniox_ws.send(b"")
                                        break
                                    # Send zeros for silence
                                    await soniox_ws.send(
                                        np.zeros_like(audio_np).tobytes()
                                    )
                                else:
                                    silence_start = None
                                    await soniox_ws.send(audio_bytes)

                    except WebSocketDisconnect:
                        _logger.info("Client WebSocket disconnected")
                    except Exception as e:
                        _logger.error(f"Error receiving from client: {e}")

                async def send_to_client():
                    """Receive transcription from Soniox and send to client"""
                    nonlocal final_tokens
                    try:
                        while True:
                            # Receive from Soniox
                            message = await asyncio.wait_for(
                                soniox_ws.recv(), timeout=30.0
                            )
                            res = json.loads(message)

                            # Handle errors from Soniox
                            if res.get("error_code") is not None:
                                error_msg = {
                                    "type": "error",
                                    "error_code": res["error_code"],
                                    "error_message": res.get(
                                        "error_message", "Unknown error"
                                    ),
                                }
                                # Only send error to client if session was not intentionally closed
                                # (avoids confusing timeout errors after user stops recording)
                                if session_closing:
                                    _logger.info(f"Session closing, ignoring Soniox error: {error_msg}")
                                else:
                                    _logger.error(f"Soniox error: {error_msg}")
                                    await client_ws.send_json(error_msg)
                                break

                            # Process tokens
                            new_final = False
                            tokens_data = []

                            for token in res.get("tokens", []):
                                if token.get("text"):
                                    tokens_data.append(
                                        {
                                            "text": token["text"],
                                            "is_final": token.get("is_final", False),
                                        }
                                    )
                                    if token.get("is_final"):
                                        final_tokens.append(token["text"])
                                        new_final = True

                            # Build full transcript
                            transcript = "".join(final_tokens)

                            # Send transcript to client when there are new final tokens
                            if new_final:
                                _logger.info(f"Transcript: {transcript}")
                                await client_ws.send_json(
                                    {
                                        "type": "transcript",
                                        "transcript": transcript,
                                        "tokens": tokens_data,
                                        "finished": False,
                                    }
                                )

                                # Schedule LLM processing with debounce
                                self._schedule_llm_processing(
                                    transcript, client_ws, client_id, final_tokens
                                )

                            # Send partial results too (for real-time feedback)
                            elif tokens_data:
                                await client_ws.send_json(
                                    {
                                        "type": "partial",
                                        "tokens": tokens_data,
                                    }
                                )

                            # Check if session finished
                            if res.get("finished"):
                                _logger.info("Soniox session finished")
                                await client_ws.send_json(
                                    {
                                        "type": "finished",
                                        "transcript": transcript,
                                        "finished": True,
                                    }
                                )
                                break

                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await client_ws.send_json({"type": "ping"})
                    except Exception as e:
                        _logger.error(f"Error sending to client: {e}")

                # Run both tasks concurrently
                await asyncio.gather(
                    receive_from_client(),
                    send_to_client(),
                    return_exceptions=True,
                )

        except WebSocketDisconnect:
            _logger.info("Client disconnected during session")
        except Exception as e:
            _logger.error(f"STT session error: {e}")
            try:
                await client_ws.send_json(
                    {
                        "type": "error",
                        "error_code": "INTERNAL_ERROR",
                        "error_message": str(e),
                    }
                )
            except:
                pass
        finally:
            # Cleanup timer for this client
            if client_id in self.llm_timers:
                self.llm_timers[client_id].cancel()
                del self.llm_timers[client_id]
            _logger.info("STT session closed")


# Global instance
stt_manager = SonioxSTTManager()
