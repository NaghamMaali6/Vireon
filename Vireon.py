'''
AI Project2: Comparative Study of Image Classification Using Decision Tree, Naive Bayes, and Feedforward Neural Networks

Vireon: is a type of small beautiful songbird — the name was chosen to reflect the bird classification theme of the project

Nagham Maali - 1212312 - Section 2
Lina Abufarha - 1211968 - Section 1
'''

# === Necessary Libraries ===
import os  #Used to interact with the operating system (like accessing files and folders)

import numpy as np  #Useful for handling numbers, arrays, and doing mathematical operations

from PIL import Image , ImageTk  #'Pillow' (PIL) helps with loading, editing, and displaying images in Python GUIs
#- Image: Used to open and process image files (resize, convert, etc.)
#- ImageTk: Converts images into a format that Tkinter can display

from tkinter import Tk , Label , Canvas , Frame , Button , StringVar  #Imports parts of Tkinter (the GUI library)
#- Tk: The main window where everything runs
#- Label: Displays text or images in the GUI
#- Canvas: Used for drawing graphics (like the splash screen)
#- Frame: A container that groups widgets together
#- Button: A clickable button
#- StringVar: A variable that holds dynamic text to be shown in the GUI

from sklearn.model_selection import train_test_split  #Splits data into training and testing sets
#- This helps test if the model is learning properly using unseen data

from sklearn.naive_bayes import GaussianNB  #A simple machine learning model (Naive Bayes classifier)
#- Works well for small datasets and is fast

from sklearn.tree import DecisionTreeClassifier  #Another model that learns by asking "yes/no" questions like a tree

import matplotlib.pyplot as plt  #'matplotlib.pyplot' is a popular plotting library in Python
#- We use it to create visualizations like charts, graphs, and figures
#- In this project, it's used to draw the Decision Tree as a visual plot
#- 'plt' is the nickname (alias) we give it so we can write shorter code (like plt.show())

from sklearn.tree import plot_tree  #'plot_tree' is a function from Scikit-learn used to visualize decision trees
#- It creates a flowchart-like diagram showing how the model splits data
#- Each node shows decisions made by the tree, and the final leaf nodes show the predicted class
#- Helps you understand how the Decision Tree model is thinking

from sklearn.neural_network import MLPClassifier  #A more advanced model that mimics the brain (Multi-layer Perceptron)

from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
#Tools to evaluate how good your model's predictions are:
#- accuracy_score: Checks how many predictions were correct
#- confusion_matrix: Shows what the model got right and wrong for each class
#- classification_report: Gives precision, recall, and F1-score for each class

import tkinter.messagebox as msgbox  #Used to show small popup message boxes in the GUI

from tkinter import Toplevel , Scrollbar , Text , filedialog , OptionMenu  #GUI elements for creating custom windows with scrollable text
#- Toplevel: Creates a new small popup window
#- Scrollbar: Lets you scroll inside a text box
#- Text: A text area where you can show long content (like classification reports)
#- filedialog: Opens a file picker dialog to let the user choose an image file from their computer
#- OptionMenu: A dropdown menu that lets the user select an option from a list (used here to pick the model)

import pandas as pd  #Powerful library for working with tabular data (like tables, reports, and CSV files)

# === Constants ===
COVER_PATH = r"C:\Users\User\Desktop\AI\Project2\CoverPage.jpg"  #The full path to the cover image that shows on the splash screen when app starts
#- The r before the string is for "raw string" so Python doesn't treat backslashes as special characters

#A dictionary that maps bird names to the path of their representative image:
BIRD_IMAGES = {"American Goldfinch": r"C:\Users\User\Desktop\AI\Project2\Birds\AmericanGoldfinch.jpg" , "Barn Owl": r"C:\Users\User\Desktop\AI\Project2\Birds\BarnOwl.jpg" , "Carmine Bee-eater": r"C:\Users\User\Desktop\AI\Project2\Birds\CarmineBee-eater.jpg" , "Downy Woodpecker": r"C:\Users\User\Desktop\AI\Project2\Birds\DownyWoodpecker.jpg" , "Emperor Penguin": r"C:\Users\User\Desktop\AI\Project2\Birds\EmperorPenguin.jpg" , "Flamingo": r"C:\Users\User\Desktop\AI\Project2\Birds\Flamingo.jpg"}
#- These are the bird images shown in the main window (intro screen)

#A dictionary to keep color codes in one place so the app has a consistent theme: 
COLORS = {"primary": "#ffd012" , "accent": "#a8b13e" , "background": "#1a1a1a" , "light": "#ffffff" , "soft": "#babb81" , "PPY": "#fff5cc"}
#- Yellow : used for main buttons and highlights
#- Olive green : used for secondary buttons
#- Dark gray : used as the app's background color
#- White : used for text or labels
#- Soft green : used for extra visual styling

DATASET_DIR = r"C:\Users\User\Desktop\AI\Project2\Bird_Speciees_Dataset"  #The full path to the main dataset folder that contains all bird class folders

IMAGE_SIZE = (64 , 64)  #The size (width, height) that every image will be resized to before it's used in the model

# === Global variables to hold data ===
X_train = X_test = y_train = y_test = class_names = None  #These variables will store the data once the dataset is loaded
#- X_train: Features used to train the model
#- X_test: Features used to test the model
#- y_train: Labels for training
#- y_test: Labels for testing
#- class_names: Names of the bird categories (folder names from dataset)


# === Load and preprocess the dataset ===
def load_dataset(status_label):  #This function loads all images from the dataset folders and prepares them for machine learning
    #- It updates the GUI status label to tell the user what's happening

    global X_train , X_test , y_train , y_test , class_names  #We use global so that we can save data in the global variables defined earlier

    X , y = [] , []  #X will store all the image data (features) and y will store the labels (bird type index)

    class_names = sorted(os.listdir(DATASET_DIR))  #Get the names of all folders (bird categories) in the dataset directory
    #- sorted() puts them in alphabetical order so the order is consistent

    status_label.set("Loading dataset...")  #Update the GUI label to tell the user that loading has started

    #Loop through each bird folder (category) one by one:
    for idx , class_name in enumerate(class_names):
        class_folder = os.path.join(DATASET_DIR , class_name)  #Create the full path to this specific class folder

        #If this isn't a folder (for example, if it’s a random file), skip it:  
        if not os.path.isdir(class_folder):
            continue

        #Loop through every image file inside this bird’s folder:
        for img_file in os.listdir(class_folder):
            try:
                img_path = os.path.join(class_folder , img_file)  #Get full path to the image

                img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)  #Open the image, convert to RGB (in case it's grayscale or RGBA), and resize it to 64x64

                X.append(np.array(img).flatten())  #Convert the image to a NumPy array, flatten it into 1D (because ML models want flat input), and add to X

                y.append(idx)  #Save the label index (e.g. 0 for first class, 1 for second...)
            except:
                continue  #If image is corrupted or can't be opened, just skip it

    #Convert X and y lists to NumPy arrays, which are needed for ML models:
    X = np.array(X)
    y = np.array(y)

    #Split data into training and testing sets:
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , stratify = y , random_state = 42)
    #- 80% training, 20% testing
    #- stratify = y ensures each class is fairly represented in both sets
    #- random_state=42 is just a seed to get consistent results every run

    print("Dataset Loaded:" , len(X_train) , "training samples," , len(X_test) , "test samples")  #Print how many samples were loaded (helpful for debugging)

    status_label.set("Dataset loaded successfully!")  #Tell the user on the GUI that everything is ready

def open_classify_window():  #Function to open a new window for classifying a bird image
    classify_win = Toplevel()  #Create a new popup window
    classify_win.title("Bird Image Classification")  #Set the window title
    classify_win.configure(bg = COLORS["background"] , padx = 20 , pady = 20)  #Set background color and padding
    classify_win.geometry("500x300")  #Set the window size
    classify_win.resizable(False , False)  #Disable resizing of the window (fixed size)

    # === Variables ===
    selected_model = StringVar(value="Naive Bayes")  #Holds the selected model from dropdown, default is "Naive Bayes"
    selected_image_path = StringVar(value = "No image selected.")  #Holds the path of the selected image file

    # === Select Image Button ===
    def pick_image():  #Function to open file dialog and update image path
        path = filedialog.askopenfilename(filetypes = [("Image Files" , "*.jpg;*.jpeg;*.png")])  #Open file picker with image filters
        if path :  
            selected_image_path.set(path)  #If user selects a file, Update the StringVar with the selected path

    Button(classify_win , text = "Browse Image" , command = pick_image , bg = COLORS["primary"] , fg = "white" , font = ("Times New Roman" , 14 , "bold")).pack(pady=(5 , 0))  #Create the "Browse Image" button

    Label(classify_win , textvariable = selected_image_path , bg = COLORS["background"] , fg = COLORS["light"] , wraplength = 350 , font = ("Times New Roman" , 10)).pack(pady = (5 , 10))  #Label to display selected image path

    # === Model Dropdown ===
    Label(classify_win , text = "Select Model:" , bg = COLORS["background"] , fg = COLORS["light"] , font = ("Times New Roman" , 11)).pack()  #Label for model selection

    OptionMenu(classify_win , selected_model , "Naive Bayes" , "Decision Tree" , "Neural Network").pack(pady = (0 , 15))  #Create dropdown menu for model selection

    # === Classify Image Function ===
    def classify_image():  #Function to perform classification
        img_path = selected_image_path.get()  #Get the selected image path
        #check If the image path is invalid or file doesn't exist:
        if not os.path.exists(img_path):
            show_custom_popup("Error" , "Please select a valid image file." , msg_type = "error")  #Show error popup if not
            return  #Exit the function

        try:
            img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)  #Open the image, convert to RGB, and resize
            img_array = np.array(img).flatten().reshape(1 , -1)  #Flatten the image into into 1D vectors(Convert image to NumPy array and reshape for prediction)

            model_name = selected_model.get()  #Get the selected model name

            if model_name == "Naive Bayes":  #If Naive Bayes is selected
                model = GaussianNB().fit(X_train , y_train)  #Train Naive Bayes model
            elif model_name == "Decision Tree":  #If Decision Tree is selected
                model = DecisionTreeClassifier(random_state = 42).fit(X_train , y_train)  #Train Decision Tree
            else:  #Neural Network
                model = MLPClassifier(hidden_layer_sizes = (100 , 50) , max_iter = 500 , random_state = 42).fit(X_train , y_train)  ## Train Neural Network

            prediction = model.predict(img_array)[0]  #Predict the bird class
            bird_name = class_names[prediction]  #Get the bird name from class label

            show_prediction_popup(bird_name)  #Show the result in a styled popup

        #Catch any unexpected errors:
        except Exception as e:
            show_custom_popup("Error" , f"Something went wrong:\n{e}" , msg_type = "error")

    # === Classify Button ===
    Button(classify_win , text = "Classify Image" , command = classify_image , bg = COLORS["primary"] , fg = "white" , font = ("Times New Roman" , 12 , "bold")).pack(pady = (10 , 0))

def show_prediction_popup(bird_name):  #Function to show a popup with the predicted bird name
    popup = Toplevel()  #Create a new top-level popup window
    popup.title("Classification Result")  #Set the title of the popup
    popup.configure(bg = COLORS["background"] , padx = 30 , pady = 20)  #Set background color and padding
    popup.geometry("400x200")  #Set the window size
    popup.resizable(False , False)  #Prevent resizing of the popup

    Label(popup , text = "Classification" , font = ("Times New Roman" , 14 , "bold") , bg = COLORS["background"] , fg = COLORS["primary"]).pack(pady = (0 , 10))  #Add a title label inside the popup

    Label(popup , text = f"The bird in the image is:\n{bird_name}" , font = ("Times New Roman" , 14 , "bold") , bg = COLORS["background"] , fg = "white" , wraplength = 300 , justify = "center").pack()  #Label to show the predicted bird name

    Button(popup , text = "OK" , command = popup.destroy , bg = COLORS["primary"] , fg = "white" , font = ("Times New Roman" , 12 , "bold")).pack(pady=(15 , 0))  #Add an "OK" button to close the popup when it's clicked

def show_custom_popup(title, message, msg_type = "info"):  #Function to show info or error popups
    popup = Toplevel()  #Create a new popup window
    popup.title(title)  #Set the window title from the passed-in argument
    popup.configure(bg = COLORS["background"] , padx = 30 , pady = 20)  #Set background color and padding
    popup.geometry("300x200")  #Set window size
    popup.resizable(False , False)  #Disable resizing

    Label(popup , text = title , font = ("Times New Roman" , 14 , "bold") , bg = COLORS["background"] , fg = "white").pack(pady = (0 , 10))  #Add a label showing the title

    Label(popup , text = message , font = ("Times New Roman" , 12) , bg = COLORS["background"] , fg = "white" , wraplength = 350 , justify = "center").pack()  #Add the main message text

    Button(popup , text = "OK" , command = popup.destroy , bg = COLORS["primary"] , fg = "white" , font = ("Times New Roman" , 12 , "bold")).pack(pady = (15 , 0))  #Add an "OK" button to close the popup when it's clicked

# === Naive Bayes model evaluation ===
def run_naive_bayes():  #This function runs the Naive Bayes model on the dataset and shows the results in a styled window
    #Check if the dataset was loaded:
    #If the training data is still None (empty), it means the user tried to run the model before loading the dataset#
    if X_train is None or y_train is None:
        show_alert_messagebox("Dataset not loaded yet!!!")  #Show a warning message box if there's no dataset
        return  #Exit the function and do nothing else

    print("\nNaive Bayes selected")  #Just a note in the console

    model = GaussianNB()  #Create a Naive Bayes classifier (Gaussian version works well with numeric data)
    model.fit(X_train , y_train)  #Train the model using training data (images and labels)
    y_pred = model.predict(X_test)  #Use the trained model to make predictions on the test data
    acc = accuracy_score(y_test , y_pred)  #Calculate the accuracy (percentage of correct predictions)
    cm = confusion_matrix(y_test , y_pred)  #Create a confusion matrix (shows how many were correctly/incorrectly classified per class)

    report_dict = classification_report(y_test , y_pred , target_names = class_names , digits = 2 , output_dict = True)  #Create a classification report (precision, recall, F1-score)
    #- output_dict=True returns it as a dictionary so we can style it

    print("Model trained and tested")
    print("\nAccuracy:" , acc)

    print("\nConfusion Matrix:\n" , cm)  #Print the raw confusion matrix (for debugging)
    print("*************************************************")

    # === Confusion Matrix ===
    col_width = max(len(name) for name in class_names) + 2  #Find the widest label name for clean formatting
    header = f"{'':<{col_width}}" + "".join(f"{name:^{col_width}}" for name in class_names)  #Build the header row (bird class names)
    matrix_rows = []  #Prepare to build each row of the matrix
    #For each row in the matrix (for each actual class):
    for i , row in enumerate(cm):
        row_str = f"{class_names[i]:<{col_width}}" + "".join(f"{val:^{col_width}}" for val in row)  #Add each row (actual class label + predictions for all classes)
        matrix_rows.append(row_str)  #Add this formatted row to the list

    cm_pretty = "📊 CONFUSION MATRIX:\n\n" + header + "\n" + "\n".join(matrix_rows)  #Combine everything into a single string to show in the GUI

    # === Format Classification Report as Table ===
    df = pd.DataFrame(report_dict).T  #Turn the report dictionary into a pandas DataFrame (for easier formatting)
    df = df[['precision' , 'recall' , 'f1-score' , 'support']]  #Keep only the important columns
    df.index = [label.upper() for label in df.index]  #Make class names all uppercase
    report_lines = []
    report_lines.append("\n\n📄 CLASSIFICATION REPORT:\n")
    report_lines.append(f"{'':<30}{'Precision':^12}{'Recall':^12}{'F1-Score':^12}{'Support':^12}\n")  #Add table headers

    #Add a row for each class or summary:
    for label in df.index:
        values = df.loc[label]
        line = f"{label:<30}{values['precision']:^12.2f}{values['recall']:^12.2f}{values['f1-score']:^12.2f}{values['support']:^12.0f}"
        report_lines.append(line)

    report_pretty = "\n".join(report_lines)  #Combine everything into one long string to display

    # === Show Results in a Styled Message Box ===
    show_custom_messagebox(title = "Naive Bayes Results" , accuracy = acc , confusion_matrix = cm_pretty , report = report_pretty , bg = "#fff5cc" , fg = "#000000" , border = "#ffd012")

def show_custom_messagebox(title , accuracy , report , confusion_matrix , bg , fg , border):  #This function creates a popup window that shows the model results in a stylish format
    #It accepts:
    #- title: the window title
    #- accuracy: the model's accuracy value
    #- report: formatted classification report (as string)
    #- confusion_matrix: formatted matrix (as string)
    #- bg: background color of the window
    #- fg: font color for main texts
    #- border: title color (usually to match the button that launched it)

    msg_win = Toplevel()  #Create a new top-level window (separate popup above main GUI)
    msg_win.title(title)  #Set the window title
    msg_win.configure(bg = bg , padx = 10 , pady = 10)  #Style the background using provided `bg` color (like light yellow) and Add some padding around the contents

    msg_win.resizable(True , True)  #Allow the popup window to be resized (horizontally/vertically)

    Label(msg_win , text = title , bg = bg , fg = border , font = ("Times New Roman" , 16 , "bold")).pack(pady=(5 , 10))  #Bold title label at the top of the message box
    #- text is the title 
    #- color matches your primary yellow style

    Label(msg_win , text = f"Accuracy: {accuracy:.2f}" , bg = bg , fg = fg , font = ("Times New Roman" , 12 , "bold")).pack()  #Accuracy label (centered just below the title)
    
    #Create a frame inside the popup to contain the scrollable results:
    text_frame = Frame(msg_win , bg = bg)
    text_frame.pack(pady = 10)
    
    #Create a horizontal scrollbar and place it in the bottom:
    scroll = Scrollbar(text_frame , orient = "horizontal")
    scroll.pack(side = "bottom" , fill = "x")
    
    #Create a big text box inside the popup:
    txt = Text(text_frame , height = 30 , width = 90 , bg = "white" , fg = "black" , font = ("Courier New" , 10) , xscrollcommand = scroll.set , wrap = "none")
    txt.pack()
    #- font = fixed-width (Courier) so tables align nicely
    #- wrap="none" disables automatic word wrapping (lets table formatting work correctly)
    #- scrollbar is linked here via xscrollcommand

    scroll.config(command = txt.xview)  #Connect the scrollbar to the text box scroll
    
    txt.insert("end" , f"{confusion_matrix}\n\n")  #Insert the formatted confusion matrix
    txt.insert("end" , report.strip())  #Insert the formatted classification report

    txt.config(state = "disabled")  #Make the text box read-only (no editing)

def show_alert_messagebox(message):  #This function shows a styled alert popup with a message (used when dataset is not loaded)
    alert = Toplevel()  #Creates a new top-level window (a popup) that appears over the main application window
    alert.title("Alert")  #Sets the title of the popup window to "Alert"
    alert.configure(bg = COLORS["background"] , padx = 20 , pady = 20)  #Sets the background color and padding 
    alert.resizable(False , False)  #Prevents the popup window from being resized (fixed size)
    
    #Creates a label widget (text) to show the alert title:
    Label(alert , text = "⚠️Dataset Error" , bg = COLORS["background"] , fg = COLORS["primary"] , font = ("Times New Roman", 14 , "bold")).pack(pady = (0 , 10))  
    #The label is placed inside the alert popup
    #The actual text displayed in the label (a warning title)
    #Background color of the label (dark)
    #Text color (yellow - your primary theme color)
    #Adds the label to the window and gives it vertical padding (bottom = 10)

    #Creates another label widget for the actual message content:
    Label(alert , text = message , bg = COLORS["background"] , fg = COLORS["light"] , font = ("Times New Roman" , 11)).pack()  #Adds the message label to the popup window
    #This label also goes in the alert popup
    #The message to display is passed as a parameter to the function
    #Background color is dark (same as popup)
    #Text color is light/white
    
    #Creates a button widget labeled "OK" that closes the popup when clicked:
    Button(alert , text = "OK" , command = alert.destroy , bg = COLORS["primary"] , fg = COLORS["background"] , font = ("Times New Roman" , 10 , "bold") , relief = "flat").pack(pady = (15 , 0))  
    #The button is placed inside the alert popup
    # When clicked, it runs the destroy() method to close the popup
    #Button background color (yellow)
    #Button text color (dark background for contrast)
    #Makes the button appear flat (no 3D border)
    #Adds the button to the window and gives it top padding of 15

# === Decision Tree model evaluation ===
def run_decision_tree():  #This function runs a Decision Tree model on the dataset and shows the results
    #Check if the dataset was loaded:
    #If the training data is still None (empty), it means the user tried to run the model before loading the dataset#
    if X_train is None or y_train is None:
        show_alert_messagebox("Dataset not loaded yet!!!")  #Show a warning message box if there's no dataset
        return  #Exit the function and do nothing else

    print("\nDecision Tree selected")  #Just a note in the console

    model = DecisionTreeClassifier(random_state = 42)  #Creates a decision tree classifier (with fixed seed for consistent results)
    model.fit(X_train , y_train)  #Trains the model on the training data
    y_pred = model.predict(X_test)  #Predicts labels on the test set

    acc = accuracy_score(y_test , y_pred)  #Accuracy of the model
    cm = confusion_matrix(y_test , y_pred)  #Confusion matrix
    report_dict = classification_report(y_test , y_pred , target_names = class_names , digits = 2 , output_dict = True)  #Detailed report

    print("Model trained and tested")
    print("\nAccuracy:" , acc)
    
    print("\nConfusion Matrix:\n" , cm)  #Print the raw confusion matrix (for debugging)
    print("*************************************************")

    # === Format Confusion Matrix ===
    col_width = max(len(name) for name in class_names) + 2  #Find the widest label name for clean formatting
    header = f"{'':<{col_width}}" + "".join(f"{name:^{col_width}}" for name in class_names)  #Build the header row (bird class names)
    matrix_rows = []  #Prepare to build each row of the matrix
    #For each row in the matrix (for each actual class):
    for i , row in enumerate(cm):
        row_str = f"{class_names[i]:<{col_width}}" + "".join(f"{val:^{col_width}}" for val in row)  #Add each row (actual class label + predictions for all classes)
        matrix_rows.append(row_str)  #Add this formatted row to the list
        
    cm_pretty = "📊 CONFUSION MATRIX:\n\n" + header + "\n" + "\n".join(matrix_rows)  #Combine everything into a single string to show in the GUI

    # === Format Classification Report ===
    df = pd.DataFrame(report_dict).T  #Turn the report dictionary into a pandas DataFrame (for easier formatting)
    df = df[['precision' , 'recall' , 'f1-score' , 'support']]  #Keep only the important columns
    df.index = [label.upper() for label in df.index]  #Make class names all uppercase
    report_lines = []
    report_lines.append("\n\n📄 CLASSIFICATION REPORT:\n")
    report_lines.append(f"{'':<30}{'Precision':^12}{'Recall':^12}{'F1-Score':^12}{'Support':^12}\n")  #Add table headers
    
    #Add a row for each class or summary: 
    for label in df.index:
        values = df.loc[label]
        line = f"{label:<30}{values['precision']:^12.2f}{values['recall']:^12.2f}{values['f1-score']:^12.2f}{values['support']:^12.0f}"
        report_lines.append(line)
    
    report_pretty = "\n".join(report_lines)  #Combine everything into one long string to display

    # === Show Results in a Styled Message Box ===
    show_custom_messagebox(title = "Decision Tree Results" , accuracy = acc , confusion_matrix = cm_pretty , report = report_pretty , bg = "#e6edc6" , fg = "#000000" , border = "#a8b13e")
    
    # === Plotting the Decision Tree Neatly ===
    fig , ax = plt.subplots(figsize = (14 , 8) , facecolor = "#e6edc6")  #Smaller figure to force compact layout
    fig.canvas.manager.set_window_title("Decision Tree")  #Window title

    plot_tree(model , ax = ax , filled = True , rounded = True , class_names = class_names , feature_names = None , impurity = False , fontsize = 7)  #Plot the tree with reduced font and layout settings

    #Reduce top space to ensure title fits, but still use layout correction:
    plt.subplots_adjust(top = 0.88)
    plt.tight_layout()
    plt.show()

# === Feedforward Neural Network model evaluation ===
def run_neural_network():
    #Check if the dataset was loaded:
    #If the training data is still None (empty), it means the user tried to run the model before loading the dataset#
    if X_train is None or y_train is None:
        show_alert_messagebox("Dataset not loaded yet!!!")  #Show a warning message box if there's no dataset
        return  #Exit the function and do nothing else

    print("\nFeedforward Neural Network selected")  #Just a note in the console

    # === Create and Train MLPClassifier ===
    model = MLPClassifier(hidden_layer_sizes=(100 , 50) , max_iter = 500 , random_state = 42)
    #- 2 hidden layers: first with 100 neurons, second with 50 -> This gives the model the ability to learn complex, non-linear patterns in the image data (after flattening the pixels).
    #- the maximum number of training cycles (epochs) the model can go through: 500 gives the model more time to learn.
    #- "random_state = 42" Ensures that every time you run the model, you get the same results (same weight initialization, same shuffling of data, etc.)
    
    model.fit(X_train , y_train)  #Train the neural network
    y_pred = model.predict(X_test)  #Test on unseen data

    acc = accuracy_score(y_test , y_pred)  #Accuracy of the model
    cm = confusion_matrix(y_test , y_pred)  #Confusion matrix
    report_dict = classification_report(y_test , y_pred , target_names = class_names , digits = 2 , output_dict = True)  #Detailed report

    print("Model trained and tested")
    print("\nAccuracy:" , acc)
    
    print("\nConfusion Matrix:\n" , cm)  #Print the raw confusion matrix (for debugging)
    print("*************************************************")

    # === Format Confusion Matrix for Display ===
    col_width = max(len(name) for name in class_names) + 2  #Find the widest label name for clean formatting
    header = f"{'':<{col_width}}" + "".join(f"{name:^{col_width}}" for name in class_names)  #Build the header row (bird class names) 
    matrix_rows = []  #Prepare to build each row of the matrix
    #For each row in the matrix (for each actual class):
    for i , row in enumerate(cm):
        row_str = f"{class_names[i]:<{col_width}}" + "".join(f"{val:^{col_width}}" for val in row)  #Add each row (actual class label + predictions for all classes)
        matrix_rows.append(row_str)  #Add this formatted row to the list

    cm_pretty = "📊 CONFUSION MATRIX:\n\n" + header + "\n" + "\n".join(matrix_rows)  #Combine everything into a single string to show in the GUI

    # === Format Classification Report ===
    df = pd.DataFrame(report_dict).T  #Turn the report dictionary into a pandas DataFrame (for easier formatting)
    df = df[['precision' , 'recall' , 'f1-score' , 'support']]  #Keep only the important columns
    df.index = [label.upper() for label in df.index]  #Make class names all uppercase
    report_lines = []
    report_lines.append("\n\n📄 CLASSIFICATION REPORT:\n")
    report_lines.append(f"{'':<30}{'Precision':^12}{'Recall':^12}{'F1-Score':^12}{'Support':^12}\n")  #Add table headers

    #Add a row for each class or summary:
    for label in df.index:
        values = df.loc[label]
        line = f"{label:<30}{values['precision']:^12.2f}{values['recall']:^12.2f}{values['f1-score']:^12.2f}{values['support']:^12.0f}"
        report_lines.append(line)

    report_pretty = "\n".join(report_lines)  #Combine everything into one long string to display

    # === Show Results in a Styled Message Box ===
    show_custom_messagebox(title = "Neural Network Results" , accuracy = acc , confusion_matrix = cm_pretty , report = report_pretty , bg = "#f0f2e2", fg = "#1a1a1a", border = COLORS["soft"])

# === Main GUI Window ===
def launch_intro_window():  #This function launches the main GUI window after the splash screen closes
    splash.destroy()  #Closes the splash screen window

    intro = Tk()  #Creates the main application window (new Tkinter root)
    intro.title("VIREON")  #Sets the window title (shown at the top of the window)
    intro.configure(bg=COLORS["background"])  #Sets the background color of the window using your color palette
    intro.geometry("700x560")  #set dimensions

    #Adds a big title label to the top of the window:
    Label(intro , text = "Bird Species Image Classifier" , fg = COLORS["primary"] , bg = COLORS["background"] , font = ("Times New Roman" , 24 , "bold") ).pack(pady = 20)  
    #Parent window where the label will appear
    #The text shown in the label
    #Text color (yellow)
    #Background color (dark gray)
    #Packs (places) the label in the window with vertical padding
    
    frame = Frame(intro, bg = COLORS["background"])  #Creates a frame to hold bird images and labels
    frame.pack()  # dds the frame to the window
    
    #Loops through each bird name and its index:
    for i , bird in enumerate(BIRD_IMAGES.keys()):  
        img_path = BIRD_IMAGES[bird]  #Gets the path of the current bird image
        
        #Checks if the image file actually exists at the path:        
        if os.path.exists(img_path):  
            img = Image.open(img_path).resize((100 , 100))  #Opens the image and resizes it to 100x100 pixels
            imgTk = ImageTk.PhotoImage(img)  #Converts the image to a format Tkinter can display
            lbl = Label(frame , image=imgTk , bg = COLORS["background"])  #Creates a label to show the image
            lbl.image = imgTk  #Keeps a reference to avoid garbage collection (Tkinter bug workaround)
            lbl.grid(row = (i // 3 * 2) , column = (i % 3) , padx = 20 , pady = 5)  #Arranges the image in a grid layout
        
        #Creates a text label below the image to display the bird name:
        Label(frame , text = bird , fg = COLORS["primary"] , bg = COLORS["background"] , font = ("Times New Roman" , 10 , "bold")).grid(row = (i // 3 * 2 + 1) , column = (i % 3))  #Arranges the name label in the grid under the image
        #Parent frame for the label
        #Text color (soft green)
        #Background color (dark)
        
    btn_frame = Frame(intro , bg = COLORS["background"])  #Creates a frame to group the 3 model buttons
    btn_frame.pack(pady = 25)  #Adds the frame to the window with vertical padding

    # === Naive Bayes Button ===
    Button(btn_frame , text = "Naive Bayes" , width = 18 , bg = COLORS["PPY"] , fg = COLORS["background"] , font = ("Times New Roman" , 12 , "bold") , command = run_naive_bayes).grid(row = 0 , column = 0 , padx = 10) 
    #Calls run_naive_bayes function when button is clicked
    
    # === Decision Tree Button ===
    Button(btn_frame , text = "Decision Tree" , width = 18 , bg = COLORS["accent"] , fg = COLORS["background"] , font = ("Times New Roman" , 12 , "bold") , command = run_decision_tree).grid(row = 0 , column = 1 , padx = 10) 

    # === Neural Network Button ===
    Button(btn_frame , text = "Feedforward Neural Network" , width = 28 , bg = COLORS["soft"] , fg = COLORS["background"] , font=("Times New Roman" , 12 , "bold") , command = run_neural_network).grid(row = 0 , column = 2 , padx = 10)
    
    # === Add Image to Classify button ===
    Button(intro , text = "Add an Image to Classify" , width = 30 , bg = COLORS["primary"] , fg = COLORS["background"] , font = ("Times New Roman" , 12 , "bold") , command = open_classify_window).pack(pady = (0 , 10))

    
    status_text = StringVar()  #Creates a special variable that updates the label text automatically
    status_text.set("Loading dataset...")  #Initial message shown to the user
    
    #Label to show status updates:
    status_label = Label(intro , textvariable = status_text , fg = COLORS["light"] , bg = COLORS["background"] , font = ("Times New Roman" , 10 , "italic"))  #Binds the label to the StringVar to allow dynamic updates
    status_label.pack(pady = (10 , 20))  #Places the label in the window

    intro.after(500 , lambda: load_dataset(status_text))  #Calls load_dataset() after 500ms, passing the status label
    intro.mainloop()  #Starts the event loop to keep the window open and interactive

# === Splash Screen ===
splash = Tk()  #Creates a new Tkinter window for the splash screen (temporary intro window)
splash.overrideredirect(True)  #Removes the window borders and title bar (makes it look cleaner and custom)

cover = Image.open(COVER_PATH).resize((900 , 600))  #Opens the image from the COVER_PATH and resizes it to 900x600
coverTk = ImageTk.PhotoImage(cover)  #Converts the image into a Tkinter-compatible format for display

canvas = Canvas(splash , width = 900 , height = 600 , highlightthickness = 0)  #Creates a canvas to hold the splash image, same size as image
canvas.pack()  #Adds the canvas to the splash window
canvas.create_image(0 , 0 , image = coverTk , anchor = "nw")  #Places the image on the canvas starting from the top-left corner (0,0)

splash.update_idletasks()  #Forces Tkinter to calculate layout and update the splash window before showing it

w , h = splash.winfo_width() , splash.winfo_height()  #Gets the current width and height of the splash screen
x = (splash.winfo_screenwidth() // 2) - (w // 2)  #Calculates X position to center the splash on the screen horizontally
y = (splash.winfo_screenheight() // 2) - (h // 2)  #Calculates Y position to center the splash on the screen vertically
splash.geometry(f"{w}x{h}+{x}+{y}")  #Applies the calculated size and position to the splash screen

splash.after(3000 , launch_intro_window)  #Tells Tkinter to wait 3000ms (3 seconds) before calling launch_intro_window()

splash.mainloop()  #Starts the splash screen's event loop so it stays visible until replaced