"""Bound unused history for forward-only Transformers SAM3 streaming.

The installed tracker reads recent non-conditioning memory/object pointers and
conditioning anchors. Preserve ALL conditioning outputs and a conservative
recent window; do not reset identity or delete object tracks. This is not for
reverse propagation, replay, or interactive correction of old frames.
"""


def new_streaming_session(**kwargs):
    from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession

    class StreamingSession(Sam3VideoInferenceSession):
        @property
        def num_frames(self):
            # The upstream tracker sometimes sees streaming=False even when
            # SAM3 receives individual frames. Its pointer lookup compares past
            # frame indices to num_frames, so this must stay the logical total,
            # never the number of currently cached frames after eviction.
            return max(self.processed_frames, default=-1) + 1 if self.processed_frames is not None else None

    return StreamingSession(**kwargs)


def trim_streaming_history(session, frame_index, window):
    cutoff = frame_index - window + 1
    if cutoff <= 0:
        return

    def trim(mapping):
        for index in list(mapping):
            if index < cutoff:
                del mapping[index]

    trim(session.processed_frames)
    for output in session.output_dict_per_obj.values():
        trim(output["non_cond_frame_outputs"])
    # A future forward-only frame is never an already tracked old frame.
    for mapping in session.frames_tracked_per_obj.values():
        trim(mapping)
    for mapping in session.point_inputs_per_obj.values():
        trim(mapping)
    for mapping in session.mask_inputs_per_obj.values():
        trim(mapping)
    trim(session.obj_id_to_tracker_score_frame_wise)
    trim(session.suppressed_obj_ids)
    # Hotstart removal is disabled in the streaming API. Keep recent metadata
    # for diagnostics; these lists otherwise grow with every missing frame.
    for mapping in (session.unmatched_frame_inds, session.overlap_pair_to_frame_inds):
        for key, indices in mapping.items():
            mapping[key] = [i for i in indices if i >= cutoff]
