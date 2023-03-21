
#import the libraries
import PIL.Image
import PIL.ImageDraw
import cv2
import face_recognition
import os

data_dir = "/home/albert/Downloads/Telegram Desktop/AkharinKhabar/photos/"


def get_all_files(directory):
    """
    this function returns all files in the directory
    :param directory:
    :return:
    """
    files = []
    filenames = []
    dir = ""
    for (dirpath, dirs, fnames) in os.walk(directory):
        dir = dirpath
        filenames = fnames
        files.extend(fnames)
        break

    f = [dir + "/" + filename for filename in files]

    return filenames, f


filenames, files = get_all_files(data_dir)
offset = 0
filenames, files = filenames[offset:], files[offset:]

for i, (fn, f) in enumerate(zip(filenames, files)):
    print("image {}: {}".format(i + 1, fn))
    img = cv2.imread(f)


    # Load the jpg file into a NumPy array
    image = face_recognition.load_image_file(f)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    number_of_faces = len(face_locations)
    print("I found {} face(s) in this photograph.".format(number_of_faces))

    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)

    for face_location in face_locations:

        # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Let's draw a box around the face
        draw = PIL.ImageDraw.Draw(pil_image)
        draw.rectangle([left, top, right, bottom], outline="red")

    # Display the image on screen
    pil_image.show()



