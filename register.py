import json
import mysql.connector
from cryptography.fernet import Fernet
import random
import cv2  # OpenCV for camera access
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

mtcnn = MTCNN()

resnet = InceptionResnetV1(pretrained='vggface2').eval()

def store_user_embeddings(user_id, embeddings,key, cursor):
    # Convert the list of embeddings to a JSON string
    json_embeddings = json.dumps(embeddings)

    # Store the JSON string in the database as a BLOB
    store_embedding_sql = "INSERT INTO user_embeddings (user_id,encrypted_embeddings, encryption_key ) VALUES (%s, %s, %s)"
    cursor.execute(store_embedding_sql, (user_id, json_embeddings, key ))

def is_credit_card_associated(card_number, db_cursor):
    query = "SELECT UserID FROM encryptedcreditcards WHERE EncryptedCardNumber = %s"
    db_cursor.execute(query, (card_number,))
    result = db_cursor.fetchone()
    return result is not None

# Function to check if a CVV is already associated with a name
def is_cvv_associated(cvv, db_cursor):
    query = "SELECT UserID FROM encryptedcreditcards WHERE EncryptedCVV = %s"
    db_cursor.execute(query, (cvv,))
    result = db_cursor.fetchone()
    return result is not None

# Function to check if an expiry date is already associated with a name
def is_expiry_date_associated(expiry_date, db_cursor):
    query = "SELECT UserID FROM encryptedcreditcards WHERE EncryptedExpiryDate = %s"
    db_cursor.execute(query, (expiry_date,))
    result = db_cursor.fetchone()
    return result is not None

def is_name_unique(name, db_cursor):
    query = "SELECT COUNT(*) FROM Users WHERE UserName = %s"
    db_cursor.execute(query, (name,))
    count = db_cursor.fetchone()[0]
    return count == 0

embeddings = []

def capture_and_process_images():
    # Initialize the camera (change the camera index or path as needed)
    camera = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Camera not found or cannot be opened.")
        return
    
    # Create a window to display the camera feed
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

     # Flag to control image capturing
    capture_images = False


    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Detect faces using MTCNN
        faces, _ = mtcnn.detect(frame)

        if faces is not None:
            # Loop through detected faces
            for face in faces:
                x, y, width, height = map(int, face)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the frame in the "Camera Feed" window
        cv2.imshow("Camera Feed", frame)

        # Check for key press events
        key = cv2.waitKey(1) & 0xFF

        # Start capturing images when 'c' is pressed
        if key == ord('c'):
            capture_images = True
            print("Capturing images...")

        # Stop capturing images and break the loop when 'q' is pressed
        elif key == ord('q'):
            print("Stopping image capture.")
            break
    
     # Process and embed the images using the FaceNet model
    images = []
    if capture_images:
        for i in range(5):  # Capture 5 images (adjust the number as needed)
            ret, frame = camera.read()
            if ret:
            # Detect faces using MTCNN
                faces, _ = mtcnn.detect(frame)

            if faces is not None:
                # Loop through detected faces
                for face in faces:
                    x, y, width, height = map(int,face)

                # Crop the detected face
                detected_face = frame[y:y+height, x:x+width]

                # Ensure the detected_face has 3 channels (convert to RGB if needed)
                if detected_face.shape[2] != 3:
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)

                # Resize the image to the expected input size
                transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160, 160)), transforms.ToTensor()])
                detected_face_tensor = transform(detected_face)

                # Generate face embeddings using InceptionResnetV1
                embedding = resnet(detected_face_tensor.unsqueeze(0)).detach().numpy()
                embeddings.append(embedding)

   # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()
    
    return embeddings

# Function to encrypt embeddings
def encrypt_embeddings(embeddings):
    # Encrypt the embeddings and store them
    encrypted_embeddings = []
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    for embedding in embeddings:
        encrypted_embedding = cipher_suite.encrypt(embedding.tobytes())
        encrypted_embeddings.append(encrypted_embedding)

    return key, encrypted_embeddings


def validate_credit_card(card_number):
    # Remove any non-digit characters
    card_number = ''.join(filter(str.isdigit, card_number))

    if not card_number:
        return False  # If no digits remain, it's not a valid number

    # Reverse the digits and convert to integers
    digits = [int(digit) for digit in card_number[::-1]]

    # Double every second digit
    doubled_digits = [digit * 2 if index % 2 == 1 else digit for index, digit in enumerate(digits)]

    # Sum the individual digits of the doubled numbers
    summed_digits = [digit if digit < 10 else digit - 9 for digit in doubled_digits]

    # Calculate the total sum
    total = sum(summed_digits)

    # The number is valid if the total is divisible by 10
    return total % 10 == 0

def validate_cvv(cvv):
    return len(cvv) == 3  # Example: CVV should be 3 digits long

def validate_expiry_date(expiry_date):
    return len(expiry_date) == 4 and expiry_date.isnumeric()  # Example: Expiry date should be in MMYY format




#INITIAL POINT

# Store the user's name in the Users table
db_connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="KushalDahiya123@",
    database="creditcarddatabase"
)
cursor = db_connection.cursor(buffered=True)
db_cursor = db_connection.cursor()


card_id = random.randint(1000, 9999)

while True:
    # Ask the user for a name
    name = input("Enter your Username: ")

    if not is_name_unique(name, db_cursor):
        print("Username is already in use. Please choose a unique Username.")
    else:
        insert_user_sql = "INSERT INTO Users (UserName) VALUES (%s)"
        cursor.execute(insert_user_sql, (name,))

        db_connection.commit()
        break

# Retrieve the generated UserID
get_user_id_sql = "SELECT UserID FROM Users WHERE UserName = %s"
cursor.execute(get_user_id_sql, (name,))
user_id = cursor.fetchone()[0]

# Ask for dummy credit card details
while True:
    # Ask for a credit card number
    card_number = input("Enter dummy credit card number: ")

    if is_credit_card_associated(card_number, db_cursor):
        print("This credit card number is already associated with a name.")
    elif not validate_credit_card(card_number):
        print("Invalid credit card number.")
    else:
        break  # Credit card is valid and not associated with a name, exit the loop

while True:
    # Ask for a CVV
    cvv = input("Enter dummy CVV: ")

    if is_cvv_associated(cvv, db_cursor):
        print("This CVV is already associated with a name.")
    elif not validate_cvv(cvv):
        print("Invalid CVV.")
    else:
        break  # CVV is valid and not associated with a name, exit the loop

while True:
    # Ask for an expiry date
    expiry_date = input("Enter dummy expiry date: ")

    if is_expiry_date_associated(expiry_date, db_cursor):
        print("This expiry date is already associated with a name.")
    elif not validate_expiry_date(expiry_date):
        print("Invalid expiry date.")
    else:
        break  # Expiry date is valid and not associated with a name, exit the loop


# Generate a secret key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)


# Store Fernet key in the database during user registration
insert_user_sql = "INSERT INTO Userkeys (UserID, FernetKey) VALUES (%s, %s)"
cursor.execute(insert_user_sql, (user_id, key.decode()))
db_connection.commit()


# Encrypt the credit card details
encrypted_card_number = cipher_suite.encrypt(card_number.encode())
encrypted_cvv = cipher_suite.encrypt(cvv.encode())
encrypted_expiry_date = cipher_suite.encrypt(expiry_date.encode())


# Store the encrypted details in the encrypted_credit_cards table
insert_credit_card_sql = "INSERT INTO encryptedcreditcards (CardID, UserID, EncryptedCardNumber, EncryptedCVV, EncryptedExpiryDate) VALUES (%s, %s, %s, %s, %s)"
values = (card_id, user_id, encrypted_card_number, encrypted_cvv, encrypted_expiry_date)
cursor.execute(insert_credit_card_sql, values)
db_connection.commit()



embeddings = capture_and_process_images()

# Encrypt user embeddings
key, encrypted_embeddings= encrypt_embeddings(embeddings)
    
#convert the list to a string
encrypted_embeddings_str = [str(embedding) for embedding in encrypted_embeddings]

# Store user embeddings in the database
store_user_embeddings(user_id, encrypted_embeddings_str,key, cursor)
db_connection.commit()

db_cursor.close()
db_connection.close()

#print("User images have been captured, embedded, encrypted, and stored in the MySQL database.")

cursor.close()
db_connection.close()
#print("Encrypted credit card details have been associated with the user in the MySQL database.")
print("Successfully Registered.")
