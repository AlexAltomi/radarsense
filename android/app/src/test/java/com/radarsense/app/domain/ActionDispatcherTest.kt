package com.radarsense.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ActionDispatcherTest {
    // Class indexes, in model order: 0 none, 1 label1, 2 label2, 3 label3, 4 label6, 5 label7.
    private val none = 0
    private val label1 = 1
    private val label2 = 2
    private val label3 = 3
    private val label6 = 4
    private val label7 = 5

    private val dispatcher = ActionDispatcher()

    @Test
    fun startsOnSlideOnePausedWithBothLightsOn() {
        assertEquals(ActionState(slide = 1, playing = false, lights = listOf(true, true)), dispatcher.state)
    }

    @Test
    fun eachGestureMapsToItsAction() {
        assertEquals(Action.PreviousSlide, dispatcher.dispatch(label2))
        assertEquals(Action.NextSlide, dispatcher.dispatch(label3))
        assertEquals(Action.PlayPause, dispatcher.dispatch(label1))
        assertEquals(Action.LightsOff, dispatcher.dispatch(label6))
        assertEquals(Action.LightsOn, dispatcher.dispatch(label7))
    }

    @Test
    fun nextSlideMovesForwardAndStopsAtTheLastSlide() {
        repeat(4) { dispatcher.dispatch(label3) }
        assertEquals(5, dispatcher.state.slide)
        dispatcher.dispatch(label3)
        dispatcher.dispatch(label3)
        assertEquals("no wrap past the last slide", 5, dispatcher.state.slide)
    }

    @Test
    fun previousSlideMovesBackAndStopsAtTheFirstSlide() {
        dispatcher.dispatch(label3)
        dispatcher.dispatch(label3)
        assertEquals(3, dispatcher.state.slide)
        dispatcher.dispatch(label2)
        assertEquals(2, dispatcher.state.slide)
        dispatcher.dispatch(label2)
        dispatcher.dispatch(label2)
        assertEquals("no wrap before the first slide", 1, dispatcher.state.slide)
    }

    @Test
    fun playPauseToggles() {
        dispatcher.dispatch(label1)
        assertEquals(true, dispatcher.state.playing)
        dispatcher.dispatch(label1)
        assertEquals(false, dispatcher.state.playing)
    }

    @Test
    fun lightsOffAndOnApplyToBothLights() {
        dispatcher.dispatch(label6)
        assertEquals(listOf(false, false), dispatcher.state.lights)
        dispatcher.dispatch(label6) // off again changes nothing
        assertEquals(listOf(false, false), dispatcher.state.lights)
        dispatcher.dispatch(label7)
        assertEquals(listOf(true, true), dispatcher.state.lights)
    }

    @Test
    fun noneDoesNothing() {
        dispatcher.dispatch(label3)
        dispatcher.dispatch(label1)
        val before = dispatcher.state
        assertNull(dispatcher.dispatch(none))
        assertEquals(before, dispatcher.state)
    }

    @Test
    fun anUnknownClassDoesNothing() {
        val before = dispatcher.state
        assertNull(dispatcher.dispatch(9))
        assertEquals(before, dispatcher.state)
    }

    @Test
    fun theMappingTableCanBeChanged() {
        val swapped = ActionDispatcher(mapOf("label2" to Action.NextSlide))
        swapped.dispatch(label2)
        assertEquals(2, swapped.state.slide)
        assertNull("label3 is not in this table", swapped.dispatch(label3))
    }
}
