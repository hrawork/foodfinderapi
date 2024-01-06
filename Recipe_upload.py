import mysql.connector
import os

# Replace these values with your MySQL database credentials
host = "localhost"
user = "root"
password = "anas"
database = "food_recipe"

# Connect to MySQL server
connection = mysql.connector.connect(
    host=host,
    user=user,
    password=password
)

# Create a cursor object to interact with the server
cursor = connection.cursor()


try:
    # Create the database if it doesn't exist
    create_database_query = f"CREATE DATABASE IF NOT EXISTS {database}"
    cursor.execute(create_database_query)

    # Switch to the specified database
    connection.database = database

    # Example: Creating a 'users' table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS recipes  (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255) NULL,
        ingredients VARCHAR(255) NULL,
        instructions VARCHAR(255) NULL
    )
    """


    # Execute the query to create the table
    cursor.execute(create_table_query)

    # Iterate through each text file in a directory
    directory_path = "./recipes"
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract dish name, ingredients, and instructions
            dish_name = os.path.splitext(filename)[0]  # Use the filename (excluding extension) as dish_name

            ingredients_index = content.find("Ingredients:")
            instructions_index = content.find("Instructions:")

            ingredients = content[ingredients_index + len("Ingredients:"):instructions_index].strip()
            instructions = content[instructions_index + len("Instructions:"):].strip()

            # Insert data into the 'recipes' table
            insert_query = "INSERT INTO recipes (title, ingredients, instructions) VALUES (%s, %s, %s)"
            data = (dish_name, ingredients, instructions)
            cursor.execute(insert_query, data)
    # Commit the changes to the database
    connection.commit()

    print("Data inserted successfully!")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the cursor and connection
    cursor.close()
    connection.close()
