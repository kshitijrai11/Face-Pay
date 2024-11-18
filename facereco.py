import json

import cv2
import mysql.connector
import numpy as np
import torch
from cryptography.fernet import Fernet
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet
from facenet_pytorch import InceptionResnetV1


resnet = InceptionResnetV1(pretrained='vggface2').eval()


"""def get_encrypted_key_from_database(user_id, db_cursor):
    try:
        select_query = "SELECT encryption_key FROM user_embeddings WHERE user_id = %s"
        db_cursor.execute(select_query, (user_id,))
        result = db_cursor.fetchone()

        if result is not None and result[0] is not None:
            return result[0]
        else:
            print(f"No encryption key found for user {user_id}")
            return None  # Handle the case where the key is not found

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None
    except Exception as e:
        print(f"Error during key retrieval: {e}")
        return None"""


"""def decrypt_user_key(encrypted_key, user_password):
    try:
        # Decode the base64-encoded encrypted key
        encrypted_key_bytes = base64.b64decode(encrypted_key)

        # Derive the encryption key using scrypt
        key = scrypt(user_password.encode(), b'salt', 32, N=2**14, r=8, p=1)

        # Extract the IV from the first 16 bytes of the encrypted key
        iv = encrypted_key_bytes[:16]

        # Initialize AES cipher in CBC mode with the derived key and IV
        cipher = AES.new(key, AES.MODE_CBC, iv)

        # Decrypt the rest of the encrypted key and unpad the result
        decrypted_key = unpad(cipher.decrypt(encrypted_key_bytes[16:]), AES.block_size)

        return decrypted_key.decode('utf-8')
    except Exception as e:
        return f"Error decrypting user key: {str(e)}"
        """

def convert_byte_string_to_array(byte_string, dtype=np.float32):
    element_size = np.dtype(dtype).itemsize
    expected_size = len(byte_string)

    # Making sure the length is a multiple of the element size
    new_length = expected_size + (element_size - expected_size % element_size) % element_size
    original_byte_string = np.frombuffer(byte_string, dtype=np.uint8).tobytes()

    adjusted_byte_string = original_byte_string.ljust(new_length, b'\0')

    try:
        # Convert the adjusted byte string back to a regular byte string
        adjusted_byte_string = adjusted_byte_string[:new_length]

        # Add print statements to display information
        
        # Convert the byte string to a NumPy array
        result = np.frombuffer(adjusted_byte_string, dtype=dtype)
        
        # Add print statements to display information after conversion
        
        return result
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise




def retrieve_user_embeddings(user_id, cursor, fernet_key):
    # Retrieve the BLOB data from the database
    get_embedding_sql = "SELECT encrypted_embeddings FROM user_embeddings WHERE user_id = %s"
    cursor.execute(get_embedding_sql, (user_id,))
    result = cursor.fetchone()

    if result is None:
        print("Encrypted embeddings not found for user.")
        return None

    json_embeddings = result['encrypted_embeddings']

    # Parse the JSON string to obtain the list of embeddings
    try:
        embeddings = json.loads(json_embeddings)
        
        return embeddings
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

    


def capture_and_process_user_face(username, cursor, fernet_key=None):
    
    
    user_id = get_user_id_by_name(username, cursor)
    stored_embedding = retrieve_user_embeddings(user_id, cursor, fernet_key)

    if stored_embedding is not None:
            # Process the user_embeddings
        print("Successfully retrieved and decrypted user embeddings.")
    else:
        print("Error retrieving or decrypting user embeddings.")
        return None, None

    if stored_embedding:
        cv2.namedWindow('Capture Face', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Capture Face', 600, 400)

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Error: Unable to open the camera.")

        while True:
            # Capture an image from the camera
            ret, frame = camera.read()

            # Display the captured frame
            cv2.imshow('Capture Face', frame)

            # Check for the 'c' key to capture the image
            key = cv2.waitKey(1)
            if key == ord('c'):
                break

            # Check for the 'q' key to quit
            elif key == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                return None, None
        camera = cv2.VideoCapture(0)
        
        
        camera.release()

        if not ret:
            raise Exception("Error: Unable to capture an image from the camera.")
        frame_tensor = preprocess_image(frame)

        frame_tensor = torch.from_numpy(frame_tensor).unsqueeze(0).float()
        # Preprocess the image (resize, normalize, etc.) based on model requirements
        

        

        captured_embedding = resnet(frame_tensor).detach().numpy().flatten()
        for item in stored_embedding:
            item_bytes = item.encode('utf-8')
            store_embeddings_array = convert_byte_string_to_array(item_bytes)
           

            # Reshape the arrays if needed
            captured_embedding = captured_embedding.reshape(1, -1)
            store_embeddings_array = store_embeddings_array.reshape(1, -1)[:,:128]

            if np.isnan(captured_embedding).any():
                print("Warning: captured_embedding contains NaN values. Handling NaN values.")
                captured_embedding = np.nan_to_num(captured_embedding)

            # Check for NaN values in decrypted_embedding
            if np.isnan(store_embeddings_array).any():
                print("Warning: decrypted_embedding contains NaN values. Handling NaN values.")
                store_embeddings_array = np.nan_to_num(store_embeddings_array)

            min_dim = min(captured_embedding.shape[1], store_embeddings_array.shape[1])

            similarity = cosine_similarity(captured_embedding[:, :min_dim], store_embeddings_array[:, :min_dim])[0][0]
            # Define a similarity threshold for considering the match as successful
            similarity_threshold = 0.9

            # Check if the similarity is above the threshold to determine if the match is successful
            if similarity > similarity_threshold:
                print("Face recognized. Payment can proceed.")
                return frame
            else:
                entered_pin = input("Face not recognized. Enter your four-digit PIN: ")
                stored_pin = '1028'
           
                stored_pin = stored_pin
                if entered_pin == stored_pin:
                    print(f"Payment is successful after PIN verification.")
                    break
                else:
                    print("Incorrect PIN. Payment failed.")
        else:
            print("Error: Unable to retrieve or process stored user embedding.")
            return None,None
 



def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

# Function to check if a username exists in the database
def username_exists(username, db_cursor):
    query = "SELECT COUNT(*) FROM Users WHERE UserName = %s"
    db_cursor.execute(query, (username,))
    # Fetch the result
    result = db_cursor.fetchone()

    if result:
        # Check if the result is a dictionary
        if isinstance(result, dict):
            count = result['COUNT(*)']
        else:
            # Assume it's a tuple
            count = result[0]

        return count > 0
    else:
        return False

# Function to retrieve the user ID based on the entered user name
def get_user_id_by_name(user_name, db_cursor):
    query = "SELECT UserID FROM Users WHERE UserName = %s"
    db_cursor.execute(query, (user_name,))
    result = db_cursor.fetchone()
    return result['UserID'] if result else None

# Function to compare user embeddings
def compare_embeddings(user_id, captured_embedding, db_cursor):
    query = "SELECT encrypted_embeddings, encryption_key FROM user_embeddings WHERE user_id = %s"
    db_cursor.execute(query, (user_id,))
    result = db_cursor.fetchone()
    
    if not result:
        return False
    
    encrypted_embeddings_str, key = result
    cipher_suite = Fernet(key)
    encrypted_embeddings = np.frombuffer(cipher_suite.decrypt(encrypted_embeddings_str))
    
    # Compare the captured embedding with the stored embedding
    # You can use a suitable similarity metric like cosine similarity
    similarity = cosine_similarity([captured_embedding], [encrypted_embeddings])[0][0]
    
    return similarity >= 0.9  # Adjust the threshold as needed


def preprocess_image(image):
    # Check if the input is a NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    # Perform any necessary preprocessing steps on the image
    # (e.g., resize, normalize, convert to the required data type)
    
    # Example: Convert image to the required size (160x160) and normalize
    image = cv2.resize(image, (160, 160))

    # Check if the image has three channels (RGB) or convert if needed
    if image.ndim == 2:
        # If the image is grayscale, convert it to three channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalize to the range [-1, 1]
    image = (image - 127.5) / 128.0

    # Convert the image to the appropriate data type (e.g., float32)
    image = image.astype(np.float32)

    # Ensure the image has the correct shape (channels, height, width)
    # If using RGB images, make sure it's in the shape (3, 160, 160)
    if image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))

    return image



# Payment process
def make_payment(username, fernet_key):
    # Get the payment amount from the user
    payment_amount = float(input("Enter the payment amount: "))

    # Process the user's face and obtain the embedding
    decrypted_embeddings = capture_and_process_user_face(username, cursor, fernet_key)

    if decrypted_embeddings is not None and len(decrypted_embeddings) > 0:
        # Payment successful
        print(f"Payment of Rs{payment_amount} successful.")
    
    
        
def is_matching_embedding(embedding1, embedding2):
    # Implement your own logic to compare the two embeddings (e.g., using a similarity threshold)
    # Return True if the embeddings match, and False if they don't
    similarity_threshold = 0.9  # You can adjust this threshold
    similarity_score = calculate_similarity(embedding1, embedding2)  # Implement this function
    return similarity_score >= similarity_threshold




def calculate_similarity(embedding1, embedding2):
    # Ensure that embedding1 and embedding2 are PyTorch tensors
    embedding1 = torch.Tensor(embedding1)
    embedding2 = torch.Tensor(embedding2)

    # Normalize the embeddings
    embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=1)
    embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=1)
   


   # Calculate the cosine similarity
    similarity_score = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)

    return similarity_score.item()



if __name__ == "__main__":

    # Connect to the database
    db_connection = mysql.connector.connect(
   host="127.0.0.1",
    user="root",
    password="KushalDahiya123@",
    database="creditcarddatabase"
    )

    cursor = db_connection.cursor(dictionary=True, buffered=True)

    # Assuming you already have the Fernet key (replace 'your_key_here' with the actual key)
    fernet_key_str = 'AJNyNH9u_hQkLPupK3fYA-xI-QAzv3U3hSBJ4vDkZnI='
    fernet_key = Fernet(fernet_key_str)
    
    # Ask for a username
    username = input("Enter a username to check: ")

    # Check if the username exists
    if username_exists(username, cursor):
        print(f"Username '{username}' exists in the database.")
        make_payment(username, fernet_key)
    else:
        print(f"Username '{username}' does not exist in the database.")

    # Close the database connections at the end
    cursor.close()
    db_connection.close()
