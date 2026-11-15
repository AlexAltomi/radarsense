package com.radarsense.app.data.net

import org.json.JSONException
import org.json.JSONObject

/** What to replay. "any" as a user means the validation users; see docs/project-rules.md. */
data class StartRequest(
    val user: String = "any",
    val gesture: String = "any",
    val loop: Boolean = true,
)

/** A JSON text message from the replay server. */
sealed interface ServerMessage {
    data class Session(val fps: Int, val classes: List<String>) : ServerMessage
    data object Stopped : ServerMessage
    data object Ended : ServerMessage
    data class Error(val message: String) : ServerMessage
    data class Unknown(val type: String) : ServerMessage
}

/**
 * The server breaks the protocol, or sends frames in a format this app cannot read. Reconnecting does not help,
 * so this is not an IOException and the stream is not retried.
 */
class StreamProtocolException(message: String, cause: Throwable? = null) : IllegalStateException(message, cause)

/** Builds and reads the JSON messages of /ws/stream. The format is described in docs/project-rules.md. */
object StreamProtocol {
    /** The replay server directly, until Caddy serves it on port 8080 in Week 8. */
    const val DEFAULT_URL = "ws://10.0.2.2:8001/ws/stream"

    private val EXPECTED_SHAPE = listOf(3, 32, 32)
    private const val EXPECTED_ENCODING = "uint8"

    fun startMessage(request: StartRequest): String = JSONObject()
        .put("type", "start")
        .put("user", request.user)
        .put("gesture", request.gesture)
        .put("loop", request.loop)
        .toString()

    fun stopMessage(): String = JSONObject().put("type", "stop").toString()

    fun parse(text: String): ServerMessage {
        val json = try {
            JSONObject(text)
        } catch (e: JSONException) {
            throw StreamProtocolException("The server sent text that is not a JSON object: ${text.take(80)}", e)
        }
        return when (val type = json.optString("type")) {
            "session" -> parseSession(json)
            "stopped" -> ServerMessage.Stopped
            "ended" -> ServerMessage.Ended
            "error" -> ServerMessage.Error(json.optString("message", "no message"))
            else -> ServerMessage.Unknown(type)
        }
    }

    private fun parseSession(json: JSONObject): ServerMessage.Session {
        try {
            val shapeJson = json.getJSONArray("shape")
            val shape = List(shapeJson.length()) { shapeJson.getInt(it) }
            val encoding = json.getString("encoding")
            if (shape != EXPECTED_SHAPE || encoding != EXPECTED_ENCODING) {
                throw StreamProtocolException(
                    "The server sends frames of shape $shape in $encoding, but the app reads $EXPECTED_SHAPE in " +
                        "$EXPECTED_ENCODING."
                )
            }
            val classesJson = json.getJSONArray("classes")
            return ServerMessage.Session(
                fps = json.getInt("fps"),
                classes = List(classesJson.length()) { classesJson.getString(it) },
            )
        } catch (e: JSONException) {
            throw StreamProtocolException("The session message is missing a field: ${e.message}", e)
        }
    }

    /** Wait before reconnect attempt number [attempt] (0 for the first): 0.5 s, 1 s, 2 s, 4 s, then 5 s. */
    fun reconnectDelayMs(attempt: Long): Long = minOf(500L shl attempt.coerceIn(0L, 4L).toInt(), 5_000L)
}
