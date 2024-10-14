import av

def close_video(container):
    """
    Close the video stream and container to avoid memory leak.

    :param container: the video container.
    """
    for video_stream in container.streams.video:
        video_stream.close(strict=False)
    container.close()

def extract_key_frames(input_video: av.container.InputContainer):
    """
    Extract key frames from the input video. If there is no keyframes in the
    video, return the first frame.

    :param input_video: input video path or container.
    :return: a list of key frames.
    """

    container = input_video

    key_frames = []
    input_video_stream = container.streams.video[0]
    ori_skip_method = input_video_stream.codec_context.skip_frame
    input_video_stream.codec_context.skip_frame = 'NONKEY'
    # restore to the beginning of the video
    container.seek(0)
    for frame in container.decode(input_video_stream):
        key_frames.append(frame)
    # restore to the original skip_type
    input_video_stream.codec_context.skip_frame = ori_skip_method

    if len(key_frames) == 0:
        # logger.warning(f'No keyframes in this video [{input_video}]. Return '
                    #    f'the first frame instead.')
        container.seek(0)
        for frame in container.decode(input_video_stream):
            key_frames.append(frame)
            break

    if isinstance(input_video, str):
        close_video(container)
    return key_frames


def get_key_frame_seconds(input_video: av.container.InputContainer):
    """
    Get seconds of key frames in the input video.
    """
    key_frames = extract_key_frames(input_video)
    ts = [float(f.pts * f.time_base) for f in key_frames]
    ts.sort()
    return ts

def extract_video_frames_uniformly(
    input_video: av.container.InputContainer,
    frame_num: int,
):
    """
    Extract a number of video frames uniformly within the video duration.

    :param input_video: input video path or container.
    :param frame_num: The number of frames to be extracted. If it's 1, only the
        middle frame will be extracted. If it's 2, only the first and the last
        frames will be extracted. If it's larger than 2, in addition to the
        first and the last frames, other frames will be extracted uniformly
        within the video duration.
    :return: a list of extracted frames.
    """
    # load the input video
    # if isinstance(input_video, str):
        # container = load_video(input_video)
    # elif isinstance(input_video, av.container.InputContainer):
    container = input_video
    # else:
    #     raise ValueError(f'Unsupported type of input_video. Should be one of '
    #                      f'[str, av.container.InputContainer], but given '
    #                      f'[{type(input_video)}].')

    input_video_stream = container.streams.video[0]
    total_frame_num = input_video_stream.frames
    if total_frame_num < frame_num:
        # logger.warning('Number of frames to be extracted is larger than the '
                    #    'total number of frames in this video. Set it to the '
                    #    'total number of frames.')
        frame_num = total_frame_num
    # calculate the frame seconds to be extracted
    duration = input_video_stream.duration * input_video_stream.time_base
    if frame_num == 1:
        extract_seconds = [duration / 2]
    else:
        step = duration / (frame_num - 1)
        extract_seconds = [step * i for i in range(0, frame_num)]

    # group durations according to the seconds of key frames
    key_frame_seconds = get_key_frame_seconds(container)
    if 0.0 not in key_frame_seconds:
        key_frame_seconds = [0.0] + key_frame_seconds
    if len(key_frame_seconds) == 1:
        second_groups = [extract_seconds]
    else:
        second_groups = []
        idx = 0
        group_id = 0
        curr_group = []
        curr_upper_bound_ts = key_frame_seconds[group_id + 1]
        while idx < len(extract_seconds):
            curr_ts = extract_seconds[idx]
            if curr_ts < curr_upper_bound_ts:
                curr_group.append(curr_ts)
                idx += 1
            else:
                second_groups.append(curr_group)
                group_id += 1
                curr_group = []
                if group_id >= len(key_frame_seconds) - 1:
                    break
                curr_upper_bound_ts = key_frame_seconds[group_id + 1]
        if len(curr_group) > 0:
            second_groups.append(curr_group)
        if idx < len(extract_seconds):
            second_groups.append(extract_seconds[idx:])

    # extract frames by their group's key frames
    extracted_frames = []
    time_base = input_video_stream.time_base
    for i, second_group in enumerate(second_groups):
        key_frame_second = key_frame_seconds[i]
        if len(second_group) == 0:
            continue
        if key_frame_second == 0.0:
            # search from the beginning
            container.seek(0)
            search_idx = 0
            curr_pts = second_group[search_idx] / time_base
            for frame in container.decode(input_video_stream):
                if frame.pts >= curr_pts:
                    extracted_frames.append(frame)
                    search_idx += 1
                    if search_idx >= len(second_group):
                        break
                    curr_pts = second_group[search_idx] / time_base
        else:
            # search from a key frame
            container.seek(int(key_frame_second * 1e6))
            search_idx = 0
            curr_pts = second_group[search_idx] / time_base
            find_all = False
            for packet in container.demux(input_video_stream):
                for frame in packet.decode():
                    if frame.pts >= curr_pts:
                        extracted_frames.append(frame)
                        search_idx += 1
                        if search_idx >= len(second_group):
                            find_all = True
                            break
                        curr_pts = second_group[search_idx] / time_base
                if find_all:
                    break
            if not find_all and frame is not None:
                # add the last frame
                extracted_frames.append(frame)

    # if the container is opened in this function, close it
    if isinstance(input_video, str):
        close_video(container)
    return extracted_frames
