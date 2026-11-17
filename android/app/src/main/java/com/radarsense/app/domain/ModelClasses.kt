package com.radarsense.app.domain

/**
 * The class names of the model, in output order. They are the neutral names from model_card.json: the real gesture
 * names are not verified yet (docs/project-rules.md), so no direction or gesture word is used here.
 * A unit test compares this list with the model card in the assets.
 */
object ModelClasses {
    val NAMES: List<String> = listOf("none", "label1", "label2", "label3", "label6", "label7")

    /** The name of class [index], or "class N" when the index is outside the list. */
    fun name(index: Int): String = NAMES.getOrNull(index) ?: "class $index"
}
