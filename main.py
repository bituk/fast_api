from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import os
import cv2
import uuid

app = FastAPI()

@app.post("/vef")

async def Vef_code(video_file: UploadFile = File(...)):
    allowed_content_types = ["video/mp4"]
    
    #checking content type
    if video_file.content_type not in allowed_content_types:
        print(video_file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file format. Only videos or Mp4 files are allowed.")
    
    #Reading contents
    contents = await video_file.read()
    # current directory
    current_directory = r"/home/babul/babul/fastapi"
    videos_directory = os.path.join(current_directory,"videos")

    #creating videos directory to store videos file
    if not os.path.exists(videos_directory):
        os.makedirs(videos_directory)
    file_location = f"{videos_directory}/{video_file.filename}"
    with open(file_location,"wb") as f:
        f.write(contents)
    # Loading videos
    video = cv2.VideoCapture(file_location)

    # checking video is loaded successfully or not
    if not video.isOpened():
        print("Error in video loading")
    else:
        print("Video loaded successfully")
    
    # Extracted_frames directory
    Extracted_frames_dir = os.path.join(current_directory,"extracted_frames")
    if not os.path.exists(Extracted_frames_dir):
        os.makedirs(Extracted_frames_dir)
    
    # Initialize frame count and timestamp variables
    frame_count = 0
    timestamp = 0
    while True:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert to milliseconds

        # Read the next frame
        success, frame = video.read()

        # Break the loop if no frame is read
        if not success:
            break

        # Generate a frame file name
        frame_name = f"{Extracted_frames_dir}/frame_{frame_count}.jpg"

        # Save the frame as an image file
        cv2.imwrite(frame_name, frame)

        # Increment the frame count
        frame_count += 1

        # Update the timestamp to the next second
        timestamp += 1
    
    #Loading model
    model_path = os.path.join(current_directory,"best1.pt")
    model = YOLO(model_path)
    # Making empty list for storing predicted_class names
    predicted_labels = []
    # Getting predicted lables from extracted image
    for i in os.listdir(Extracted_frames_dir):
        # Image
        im = os.path.join(Extracted_frames_dir,i)
        # Inference
        results = model.predict(im,verbose=False,conf=0.80)

        for i in list(map(lambda x:int(x),results[0].boxes.cls.tolist())):
            labels = model.names[i]
            if labels not in predicted_labels:
                predicted_labels.append(model.names[i])
    
    # For removing stored file
    for i in os.listdir(Extracted_frames_dir):
        os.remove(os.path.join(Extracted_frames_dir,i))
    print(predicted_labels)

    mapped_part_item = {"Good Headlight":1,"Good Backlight":1,"Good Mirror":1,"Good Seat":1,"Good indicator":1,"Bad Headlight":0,"Bad Backlight":0,"Bad Mirror":0,"Bad Seat":0,"Bad indicator":0}
    # Making a function for getting parts code
    def parts_code(n):
        headlight_code = 0
        backlight_code = 0
        indicator_code = 0
        mirror_code = 0
        seat_code = 0
        predicted_list = n
        
        # For headlight
        if "Good Headlight" in predicted_list:
            if "Bad Headlight" in predicted_list:
                headlight_code = mapped_part_item["Bad Headlight"]
            else:
                headlight_code = mapped_part_item["Good Headlight"]
            
        # For backlight
        if "Good Backlight" in predicted_list:
            if "Bad Backlight" in predicted_list:
                backlight_code = mapped_part_item["Bad Backlight"]
            else:
                backlight_code = mapped_part_item["Good Backlight"]

        # For indicator
        if "Good indicator" in predicted_list:
            if "Bad indicator" in predicted_list:
                indicator_code = mapped_part_item["Bad indicator"]
            else:
                indicator_code = mapped_part_item["Good indicator"]

        # For Seat
        if "Good Seat" in predicted_list:
            if "Bad Seat" in predicted_list:
                seat_code = mapped_part_item["Bad Seat"]
            else:
                seat_code = mapped_part_item["Good Seat"]

        # For mirror
        if "Good Mirror" in predicted_list:
            if "Bad Mirror" in predicted_list:
                mirror_code = mapped_part_item["Bad Mirror"]
            else:
                mirror_code = mapped_part_item["Good Mirror"]
            
        #result_code
        result_code = str(headlight_code)+str(backlight_code)+str(indicator_code)+str(seat_code)+str(mirror_code)
        return result_code
    # For dent detection
    # Initialize frame count and timestamp variables
    frame_count = 0
    timestamp = 0
    while True:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000*8)  # Convert to milliseconds and getting 1 image per 8 second

        # Read the next frame
        success, frame = video.read()

        # Break the loop if no frame is read
        if not success:
            break

        # Generate a frame file name
        frame_name = f"{Extracted_frames_dir}/frame_{frame_count}.jpg"

        # Save the frame as an image file
        cv2.imwrite(frame_name, frame)

        # Increment the frame count
        frame_count += 1

        # Update the timestamp to the next second
        timestamp += 1
    # Making empty list for storing predicted_class names
    predicted_dent_labels = []
    # Getting predicted lables from extracted image
    for i in os.listdir(Extracted_frames_dir):
        # Image
        im = os.path.join(Extracted_frames_dir,i)
        # Inference
        results = model(im,verbose=False)
        for i in list(map(lambda x:int(x),results[0].boxes.cls.tolist())):
            labels = model.names[i]
            if labels.lower()=="dents" or labels.lower()=="scratch":
                predicted_dent_labels.append(labels)   
    Dents_count = len(predicted_dent_labels)
    # Number of dent should be less than 10
    if Dents_count >= 10:
        Dents_count = 9
    # For removing stored extracted file
    for i in os.listdir(Extracted_frames_dir):
        os.remove(os.path.join(Extracted_frames_dir,i))
    print("The Number of dents and scatch are",predicted_dent_labels.count("Dents"))
    # VEF code extracter

    def vef(predicted_class,dent_count):
        VEF_code = ""
        number_of_dents = dent_count
        n=predicted_class
        VEF_code = str(number_of_dents)+ parts_code(n)
        
        '''
        VEF code position status-
        1st- Gives number of dents
        2nd- Gives the status of headlight
        3rd - Gives the status of backlight
        4th - Gives the status of indicator
        5th - Gives the status of seat
        6th - Gives the status of side mirror
        '''
        return VEF_code
    VEF_code= vef(predicted_labels,Dents_count)
    #Releasing video or clsoing video
    video.release()
    print(VEF_code)

    return {"VEF_code":VEF_code}

@app.post("/code_video")
async def Vef_code_video(video_file: UploadFile = File(...)):
    allowed_content_types = ["video/mp4"]
    
    #checking content type
    if video_file.content_type not in allowed_content_types:
        print(video_file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file format. Only videos or Mp4 files are allowed.")
    
    #Reading contents
    contents = await video_file.read()
    # current directory
    current_directory = r"/home/babul/babul/fastapi"
    videos_directory = os.path.join(current_directory,"videos")

    #creating videos directory to store videos file
    if not os.path.exists(videos_directory):
        os.makedirs(videos_directory)
    file_location = f"{videos_directory}/{video_file.filename}"
    with open(file_location,"wb") as f:
        f.write(contents)
    # Loading videos
    video = cv2.VideoCapture(file_location)

    # checking video is loaded successfully or not
    if not video.isOpened():
        print("Error in video loading")
    else:
        print("Video loaded successfully")
    
    # Extracted_frames directory
    Extracted_frames_dir = os.path.join(current_directory,"extracted_frames")
    if not os.path.exists(Extracted_frames_dir):
        os.makedirs(Extracted_frames_dir)
    
    # Initialize frame count and timestamp variables
    frame_count = 0
    timestamp = 0
    while True:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert to milliseconds

        # Read the next frame
        success, frame = video.read()

        # Break the loop if no frame is read
        if not success:
            break

        # Generate a frame file name
        frame_name = f"{Extracted_frames_dir}/frame_{frame_count}.jpg"

        # Save the frame as an image file
        cv2.imwrite(frame_name, frame)

        # Increment the frame count
        frame_count += 1

        # Update the timestamp to the next second
        timestamp += 1
    
    #Loading model
    model_path = os.path.join(current_directory,"best1.pt")
    model = YOLO(model_path)
    # Making empty list for storing predicted_class names
    predicted_labels = []
    # Getting predicted lables from extracted image
    for i in os.listdir(Extracted_frames_dir):
        # Image
        im = os.path.join(Extracted_frames_dir,i)
        # Inference
        results = model.predict(im,verbose=False,conf=0.80)

        for i in list(map(lambda x:int(x),results[0].boxes.cls.tolist())):
            labels = model.names[i]
            if labels not in predicted_labels:
                predicted_labels.append(model.names[i])
    
    # For removing stored file
    for i in os.listdir(Extracted_frames_dir):
        os.remove(os.path.join(Extracted_frames_dir,i))
    print(predicted_labels)

    mapped_part_item = {"Good Headlight":1,"Good Backlight":1,"Good Mirror":1,"Good Seat":1,"Good indicator":1,"Bad Headlight":0,"Bad Backlight":0,"Bad Mirror":0,"Bad Seat":0,"Bad indicator":0}
    # Making a function for getting parts code
    def parts_code(n):
        headlight_code = 0
        backlight_code = 0
        indicator_code = 0
        mirror_code = 0
        seat_code = 0
        predicted_list = n
        
        # For headlight
        if "Good Headlight" in predicted_list:
            if "Bad Headlight" in predicted_list:
                headlight_code = mapped_part_item["Bad Headlight"]
            else:
                headlight_code = mapped_part_item["Good Headlight"]
            
        # For backlight
        if "Good Backlight" in predicted_list:
            if "Bad Backlight" in predicted_list:
                backlight_code = mapped_part_item["Bad Backlight"]
            else:
                backlight_code = mapped_part_item["Good Backlight"]

        # For indicator
        if "Good indicator" in predicted_list:
            if "Bad indicator" in predicted_list:
                indicator_code = mapped_part_item["Bad indicator"]
            else:
                indicator_code = mapped_part_item["Good indicator"]

        # For Seat
        if "Good Seat" in predicted_list:
            if "Bad Seat" in predicted_list:
                seat_code = mapped_part_item["Bad Seat"]
            else:
                seat_code = mapped_part_item["Good Seat"]

        # For mirror
        if "Good Mirror" in predicted_list:
            if "Bad Mirror" in predicted_list:
                mirror_code = mapped_part_item["Bad Mirror"]
            else:
                mirror_code = mapped_part_item["Good Mirror"]
            
        #result_code
        result_code = str(headlight_code)+str(backlight_code)+str(indicator_code)+str(seat_code)+str(mirror_code)
        return result_code
    # For dent detection
    # Initialize frame count and timestamp variables
    frame_count = 0
    timestamp = 0
    while True:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000*8)  # Convert to milliseconds and getting 1 image per 8 second

        # Read the next frame
        success, frame = video.read()

        # Break the loop if no frame is read
        if not success:
            break

        # Generate a frame file name
        frame_name = f"{Extracted_frames_dir}/frame_{frame_count}.jpg"

        # Save the frame as an image file
        cv2.imwrite(frame_name, frame)

        # Increment the frame count
        frame_count += 1

        # Update the timestamp to the next second
        timestamp += 1
    # Making empty list for storing predicted_class names dent and scratch
    predicted_dent_labels = []

    # Getting predicted lables from extracted image
    for i in os.listdir(Extracted_frames_dir):
        # Image
        im = os.path.join(Extracted_frames_dir,i)
        # Inference
        results = model(im,verbose=False,conf=0.1)
        for i in list(map(lambda x:int(x),results[0].boxes.cls.tolist())):
            labels = model.names[i]
            if (labels.lower()=="dents") or (labels.lower()=="scratch"):
                predicted_dent_labels.append(labels)

    Dents_count = len(predicted_dent_labels)
    # Number of dent should be less than 10
    if Dents_count >= 10:
        Dents_count = 9
    # For removing stored extracted file
    for i in os.listdir(Extracted_frames_dir):
        os.remove(os.path.join(Extracted_frames_dir,i))
    print("The Number of dents and scatch are",Dents_count)
    # VEF code extracter

    def vef(predicted_class,dent_count):
        VEF_code = ""
        number_of_dents = dent_count
        n=predicted_class
        VEF_code = str(number_of_dents)+ parts_code(n)
        
        '''
        VEF code position status-
        1st- Gives number of dents
        2nd- Gives the status of headlight
        3rd - Gives the status of backlight
        4th - Gives the status of indicator
        5th - Gives the status of seat
        6th - Gives the status of side mirror
        '''
        return VEF_code
    VEF_code= vef(predicted_labels,Dents_count)
    def predict_video(video_path):
        annotated_video_path = os.path.join(current_directory,"annotated_videos")
        if not os.path.exists(annotated_video_path):
            os.makedirs(annotated_video_path)
        filename = os.path.basename(file_location)
        base_name, extension = os.path.splitext(filename)
        
        # Read the input video file
        video = cv2.VideoCapture(video_path)
        
        # Get the video's frame rate, width, and height
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize the video writer with the same frame rate, width, and height as the input video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = f"{annotated_video_path}/{base_name}.mp4"
        print("output path",output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process the video frame by frame
        while True:
            # Read the next frame from the video
            ret, frame = video.read()
            
            # Break the loop if we have reached the end of the video
            if not ret:
                break
            
            # Perform object detection on the frame using the yolov8 model
            results = model.predict(frame,conf=0.7,verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
        # Release the video and video writer objects
        out.release()
        
        # Return the path of the output video file
        print("The annotated video is saved in ",output_path)

    #saving annotated videos
    predict_video(file_location)
    #Releasing video or clsoing video
    video.release()

    return {"VEF_code":VEF_code}