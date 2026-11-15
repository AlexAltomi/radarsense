package com.radarsense.app.data.net

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class StreamProtocolTest {
    @Test
    fun startMessageHasTheFieldsTheServerReads() {
        val json = JSONObject(StreamProtocol.startMessage(StartRequest(user = "user3", gesture = "label2", loop = false)))
        assertEquals("start", json.getString("type"))
        assertEquals("user3", json.getString("user"))
        assertEquals("label2", json.getString("gesture"))
        assertEquals(false, json.getBoolean("loop"))
    }

    @Test
    fun stopMessage() {
        assertEquals("stop", JSONObject(StreamProtocol.stopMessage()).getString("type"))
    }

    @Test
    fun parsesTheSessionMessageOfTheReplayServer() {
        // Same shape as replay.server.session_message().
        val text = """{"type": "session", "fps": 33, "shape": [3, 32, 32], "encoding": "uint8",
            "classes": ["none", "label1", "label2", "label3", "label6", "label7"]}"""
        val session = StreamProtocol.parse(text) as ServerMessage.Session
        assertEquals(33, session.fps)
        assertEquals(listOf("none", "label1", "label2", "label3", "label6", "label7"), session.classes)
    }

    @Test(expected = StreamProtocolException::class)
    fun refusesAnotherFrameShape() {
        StreamProtocol.parse("""{"type":"session","fps":33,"shape":[3,64,32],"encoding":"uint8","classes":[]}""")
    }

    @Test(expected = StreamProtocolException::class)
    fun refusesAnotherEncoding() {
        StreamProtocol.parse("""{"type":"session","fps":33,"shape":[3,32,32],"encoding":"float16","classes":[]}""")
    }

    @Test(expected = StreamProtocolException::class)
    fun refusesASessionWithoutClasses() {
        StreamProtocol.parse("""{"type":"session","fps":33,"shape":[3,32,32],"encoding":"uint8"}""")
    }

    @Test(expected = StreamProtocolException::class)
    fun refusesTextThatIsNotJson() {
        StreamProtocol.parse("hello")
    }

    @Test
    fun parsesTheOtherServerMessages() {
        assertEquals(ServerMessage.Stopped, StreamProtocol.parse("""{"type":"stopped"}"""))
        assertEquals(ServerMessage.Ended, StreamProtocol.parse("""{"type":"ended"}"""))
        assertEquals(
            ServerMessage.Error("Unknown user 'user99'."),
            StreamProtocol.parse("""{"type":"error","message":"Unknown user 'user99'."}"""),
        )
        assertEquals(ServerMessage.Unknown("hello"), StreamProtocol.parse("""{"type":"hello"}"""))
    }

    @Test
    fun reconnectDelayGrowsThenStaysAt5Seconds() {
        assertEquals(
            listOf(500L, 1_000L, 2_000L, 4_000L, 5_000L, 5_000L),
            listOf(0L, 1L, 2L, 3L, 4L, 50L).map { StreamProtocol.reconnectDelayMs(it) },
        )
    }
}
