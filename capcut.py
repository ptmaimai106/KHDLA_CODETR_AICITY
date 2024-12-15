import cv2
import os

# Input and output folder paths
input_folder = './data/'  # Folder containing the input videos
output_folder = './data/converted_videos/'  # Folder to save converted videos

# Desired output properties
desired_fps = 10
output_width = 1920
output_height = 1080

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process a single video
def convert_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get the original frame rate
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {input_video_path} with original FPS: {original_fps}")

    # Move to the 7th second
    start_frame_number = int(original_fps * 7)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    # Calculate the total number of frames for 20 seconds at the desired fps
    output_frame_count = int(desired_fps * 20)

    # Get the codec of the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4

    # Create VideoWriter object to write the output video
    out = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (output_width, output_height))

    # Frame counter for output video
    output_frames_written = 0

    # Read and process the input video frames
    while output_frames_written < output_frame_count:
        ret, frame = cap.read()

        if not ret:
            print(f"End of video reached before completing 20 seconds for {input_video_path}.")
            break

        # Resize the frame to the desired resolution
        resized_frame = cv2.resize(frame, (output_width, output_height))

        # Write the frame to the output video
        out.write(resized_frame)

        output_frames_written += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Conversion completed for {input_video_path}. Saved to {output_video_path}.")

# Iterate over all video files in the input folder
for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add any other video formats as needed
        input_video_path = os.path.join(input_folder, video_file)
        output_video_path = os.path.join(output_folder, f"converted-{video_file}")

        # Convert the video
        convert_video(input_video_path, output_video_path)

print("All videos have been processed.")
