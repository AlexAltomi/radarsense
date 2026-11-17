package com.radarsense.app.domain

/** Things a fired gesture can do on the Actions screen. */
enum class Action { PreviousSlide, NextSlide, PlayPause, LightsOff, LightsOn }

/** What the Actions screen shows. A new object is made for every change, so it is safe to share between threads. */
data class ActionState(
    /** Current slide, 1 to [ActionDispatcher.SLIDE_COUNT]. */
    val slide: Int = 1,
    val playing: Boolean = false,
    /** One entry per light. */
    val lights: List<Boolean> = List(ActionDispatcher.LIGHT_COUNT) { true },
)

/**
 * Turns a fired gesture into a change of [ActionState]. Plain Kotlin, no Android classes.
 *
 * The mapping from gesture to action is the one table [DEFAULT_MAPPING]. To change what a gesture does, change a
 * line there. The keys are the neutral class names from the model card (see [ModelClasses]).
 *
 * Not thread-safe: call it from one thread (the view model calls it on the main thread).
 */
class ActionDispatcher(private val mapping: Map<String, Action> = DEFAULT_MAPPING) {
    var state: ActionState = ActionState()
        private set

    /**
     * Applies the gesture with this class index. Returns the action that was taken, or null when the class has no
     * action ("none" or an unknown class), in which case nothing changes.
     */
    fun dispatch(classIndex: Int): Action? {
        val action = mapping[ModelClasses.name(classIndex)] ?: return null
        state = when (action) {
            Action.PreviousSlide -> state.copy(slide = (state.slide - 1).coerceAtLeast(1))
            Action.NextSlide -> state.copy(slide = (state.slide + 1).coerceAtMost(SLIDE_COUNT))
            Action.PlayPause -> state.copy(playing = !state.playing)
            Action.LightsOff -> state.copy(lights = List(LIGHT_COUNT) { false })
            Action.LightsOn -> state.copy(lights = List(LIGHT_COUNT) { true })
        }
        return action
    }

    companion object {
        const val SLIDE_COUNT = 5
        const val LIGHT_COUNT = 2

        /** The one table: which gesture does what. The gesture names are neutral because the real names are not verified. */
        val DEFAULT_MAPPING: Map<String, Action> = mapOf(
            "label2" to Action.PreviousSlide,
            "label3" to Action.NextSlide,
            "label1" to Action.PlayPause,
            "label6" to Action.LightsOff,
            "label7" to Action.LightsOn,
        )
    }
}
