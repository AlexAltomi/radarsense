package com.radarsense.app.data.net

import kotlinx.coroutines.flow.Flow

/**
 * Where the frames come from. The app uses [FrameStreamClient] (the replay server). Tests can use a fake that
 * returns a prepared list of events.
 *
 * Collecting the flow starts the stream, and cancelling the collection stops it. The flow fails with an
 * IOException when the connection is lost; see [FrameStreamClient] for the details.
 */
interface FrameSource {
    fun stream(url: String, start: StartRequest): Flow<StreamEvent>
}
