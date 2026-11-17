package com.radarsense.app.domain

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class ModelClassesTest {
    @Test
    fun namesMatchTheModelCardInTheAssets() {
        // Unit tests run in the app module folder, so this path is relative to android/app.
        val card = JSONObject(File("src/main/assets/model_card.json").readText())
        val classes = card.getJSONArray("classes")
        val fromCard = List(classes.length()) { classes.getString(it) }
        assertEquals(fromCard, ModelClasses.NAMES)
        assertEquals(GestureClassifier.NUM_CLASSES, ModelClasses.NAMES.size)
    }

    @Test
    fun anIndexOutsideTheListGetsAPlainName() {
        assertEquals("label3", ModelClasses.name(3))
        assertEquals("class 9", ModelClasses.name(9))
    }
}
