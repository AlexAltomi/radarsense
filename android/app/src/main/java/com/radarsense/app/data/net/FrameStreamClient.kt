package com.radarsense.app.data.net

import com.radarsense.app.domain.Frame
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString

/** Everything the stream delivers, in arrival order. */
sealed interface StreamEvent {
    /** The socket is open and the start request was sent. */
    data object Connected : StreamEvent
    data class Message(val message: ServerMessage) : StreamEvent
    class FrameReceived(val frame: Frame) : StreamEvent
}

/**
 * Connects to the replay server's /ws/stream and turns it into a [Flow].
 *
 * Collecting the flow opens the socket and sends the start request. Cancelling the collection sends "stop" and
 * closes the socket. The flow fails with an [IOException] when the connection is lost, so the caller can retry
 * (a reconnect sends "start" again, and the first frame after it is a clip start). It fails with
 * [StreamProtocolException] or [FrameFormatException] when the server sends something the app cannot read; retrying
 * does not help then.
 *
 * Frames are parsed on OkHttp's reader thread and the flow runs on Dispatchers.IO, never on the main thread.
 */
class FrameStreamClient(private val http: OkHttpClient = defaultHttpClient()) {

    fun stream(url: String, start: StartRequest): Flow<StreamEvent> = callbackFlow {
        val listener = object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(StreamProtocol.startMessage(start))
                trySend(StreamEvent.Connected)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    trySend(StreamEvent.Message(StreamProtocol.parse(text)))
                } catch (e: StreamProtocolException) {
                    close(e)
                }
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                try {
                    trySend(StreamEvent.FrameReceived(FrameParser.parse(bytes.toByteArray(), System.nanoTime())))
                } catch (e: FrameFormatException) {
                    close(e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(NORMAL_CLOSURE, null)
                close(IOException("The server closed the connection (code $code${if (reason.isEmpty()) "" else ", $reason"})."))
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                close(t as? IOException ?: IOException(t.message, t))
            }
        }

        val socket = http.newWebSocket(Request.Builder().url(url).build(), listener)
        awaitClose {
            socket.send(StreamProtocol.stopMessage())
            socket.close(NORMAL_CLOSURE, "client stopped")
        }
    }
        // If the consumer falls behind, old frames are dropped. FrameBuffer notices the sequence gap and empties
        // itself, so a window never contains a hole.
        .buffer(capacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST)
        .flowOn(Dispatchers.IO)

    companion object {
        private const val NORMAL_CLOSURE = 1000

        fun defaultHttpClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS) // the stream can be idle; pings detect a dead connection
            .pingInterval(10, TimeUnit.SECONDS)
            .build()
    }
}
