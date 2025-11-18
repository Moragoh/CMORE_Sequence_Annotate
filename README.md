Aren't LLMs great?

# **Ground Truth Video Annotator**

This tool is used to annotate "Success" sequences in videos, such as a block landing in a target zone. It uses a live YOLO model to draw the target zone on the video as you annotate.

## **1\. Setup & Installation**

### **Step 1: Install Dependencies**

Open your terminal and run the following command to install all required libraries:  
pip install ultralytics torch torchvision pandas numpy opencv-python Pillow

### **Step 2: File Structure**

Place your files in a single folder as shown below. The script **must** be in the same directory as keypoint\_detector.py and expects the model to be named best\_model.pt.  
/your\_project\_folder  
  ├── annotate\_video.py       \<-- The main script  
  ├── keypoint\_detector.py    \<-- Your detector class file  
  ├── best\_model.pt           \<-- Your trained YOLO model  
  └── my\_video.mp4            \<-- The video you want to annotate

## **2\. How to Use**

### **Step 1: Run the Tool**

Open your terminal, navigate to your project folder, and run the script. You **must** specify the video and the handedness (--R or \--L).  
**Example for Right-Handed:**  
python annotate\_video.py \--video my\_video.mp4 \--R

**Example for Left-Handed:**  
python annotate\_video.py \--video my\_video.mp4 \--L

#### **Command-Line Arguments:**

* **\--video**: (Required) The path to your video file.  
* **\--R or \--L**: (Required) You must choose one to set the target compartment (Right or Left).  
* **\--output**: (Optional) Specify a CSV name. By default, it saves as \[video\_name\]\_sequence\_annotations.csv.

### **Step 2: Use the Controls**

A video window will open. Use these keys to annotate:

| Key | Function | Description |
| :---- | :---- | :---- |
| **k** | **Next Frame** | Advance video by 1 frame. |
| **j** | **Prev Frame** | Rewind video by 1 frame. |
| **1** | **Mark START** | Marks the beginning of a success attempt. |
| **2** | **Mark STOP** | Marks the end of the attempt and saves the pair. |
| **q** | **Quit** | Saves all annotations to the CSV and exits. |

## **3\. Output**

The script automatically generates a CSV file (e.g., my\_video\_sequence\_annotations.csv) in your project folder with Start Frame and End Frame columns.
