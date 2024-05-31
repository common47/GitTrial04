import streamlit as st
import cv2

video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

if video_data:
    # save uploaded video to disc
    temp_file_to_save = 'c:/temp_file_1.mp4'
    write_bytesio_to_file(temp_file_to_save, video_data)
    # read it with cv2.VideoCapture(), so now we can process it with OpenCV functions
    cap = cv2.VideoCapture(temp_file_to_save)
    # grab some parameters of video to use them for writing a new, processed video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    st.write(width, height, frame_fps)
    # specify a writer to write a processed video to a disk frame by frame
    temp_file_result = 'c:/temp_file_2.mp4'
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'XVID')
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height))
    # loop though a video, process each frame and save it to a disk
    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            st.write("Can't receive frame (stream end?). Exiting ...")
            break
        # some video processing here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # write a processed frame to the video on a disk
        out_mp4.write(gray)

    # when video is fully saved to disk, open it as BytesIO and play with st.video()
    result_video = open(temp_file_result, "rb")
    st.video(result_video)
