from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import os
from termcolor import cprint
import time
import pickle
from Utils import Utils
from collections import Counter
import pyttsx3
SCALE=0.7



class Interface:
    def __init__(self, demo=False):
        #print(demo)
        self.demo = demo
        
        self.root = Tk()
        self.root.title("Projet Twizy Barbet, Breton, Coq, Mimouni, Traore")
        self.root.geometry("1000x900")

        # Création d'un Label pour le titre
        title_label = Label(self.root, text="Projet Twizy", font=("Arial", int(50*SCALE)))
        under_title_label = Label(self.root, text="Barbet, Breton, Coq, Mimouni, Traore", font=("Courier", int(20*SCALE)))
        title_label.pack(pady=int(20*SCALE))
        under_title_label.pack(pady=int(20*SCALE))
        #explication_title_label.pack(pady=int(20*SCALE))

        # Création du bouton "Choisissez une photo ou une vidéo à analyser"
        browse_button_photo = Button(self.root, text="Choisissez une photo\n ou une vidéo à analyser",font=("Arial", 13), command=self.browse_file, bg="pale green")
        browse_button_photo.pack(padx=10, pady=10)
        browse_button_photo.config(width=25, height=3)

        # Création du bouton "Lancement de la reconnaissance"
        self.launch_button = Button(self.root, text="Lancement de\n la reconnaissance",font=("Arial", 13), command=self.launch_recognition, bg="light blue")
        # Cacher le bouton au départ
        self.launch_button.pack_forget()

        self.canvas = Canvas(self.root, width=0, height=0)
        self.canvas.pack_forget()

        #création d'un label vide qui deviendra la phrase qui annonce les panneaux lors de la détection
        self.label = Label(self.root, text=" ", font=("Arial", 14))
        self.label.pack()
        
        self.engine = pyttsx3.init()

        

        # Initialisation à 0 de self.isImage et self.isVideo
        self.isImage = False #Si self.isImage = True, l'utilisateur a sélectionné une image
        self.isVideo = False #Si self.isVideo = True, l'utilisateur a sélectionné une vidéo

        data_dir = os.path.join(os.getcwd(), "data") #os.getcwd() le chemin vers le répertoire où se trouve le script en cours d'exécution
        self.reference_images_dict = {}
        self.reference_image_pre_processed_dict = {}
        
        
        for file in os.listdir(data_dir): #parcours de data_dir en ignorant les dossiers
            if os.path.isdir(os.path.join(data_dir, file)):
                continue
            
            self.reference_images_dict[os.path.join(data_dir, file)] = cv2.imread(os.path.join(data_dir, file))

            with open(os.path.join("data/pre_processed", file + ".pickle"), "rb") as f:
                self.reference_image_pre_processed_dict[os.path.join(data_dir, file)] = pickle.load(f)


        self.boundaries = [
            ([0, 110, 0], [6, 255, 255]),
            ([170, 110, 0], [255, 255, 255])
        ]

    def browse_file(self): #appelée lorsque l'utilisateur clique sur le bouton "Choisissez une photo à analyser"
        self.isImage = False
        self.isVideo = False
        cv2.destroyAllWindows()

        self.canvas.delete("all") #efface tout le contenu de la fenêtre

        self.file_path = filedialog.askopenfilename() #pour ouvrir une fenêtre de dialogue de sélection de fichier
        print("Selected file path:", self.file_path)

        # CAS VIDEO #
        if self.file_path.endswith(".avi") or self.file_path.endswith(".mp4"):
            # afficher la première image de la vidéo sélectionnée : 
            video = cv2.VideoCapture(self.file_path)
            ret, frame = video.read() # ret est la valeur de retour pour s'assurer que le frame a été correctement lu
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.release()
            
            if ret: # si le frame a été correctement lu
                image = Image.fromarray(frame) #fromarray() convertit 'frame' pour qu'il puisse être affiché sur le canevas
                self.isVideo = True #car une vidéo a été sélectionnée avec succès
            else:
                raise ValueError("No image found !")
        
        # CAS PHOT0 #
        else:
            # Chargement de l'image sélectionnée
            image = Image.open(self.file_path)
            self.isImage = True #car une photo a été sélectionnée avec succès
        
        # Redimensionnement de l'image pour l'affichage dans la fenêtre
        if image.size[1] > 500:
            size = (image.size[0] // 2, image.size[1] // 2)
            image = image.resize(size, Image.ANTIALIAS) #ANTIALIAS pour avoir une image de meilleure qualité
        
        # Conversion de l'image en format compatible avec Tkinter
        image_tk = ImageTk.PhotoImage(image)


        self.canvas.config(width=image.size[0], height=image.size[1]) #configurer la taille du canevas en fonction de la taille de l'image
        self.canvas.create_image(0, 0, anchor=NW, image=image_tk) #créer une image sur le canevas à partir de l'image chargée (et positionnement)
        self.canvas.image = image_tk
        self.canvas.pack()

        # Affichage du bouton "Lancement de la reconnaissance"
        self.launch_button.pack(padx=10, pady=10)

    def launch_recognition(self):
        if self.isVideo:
            print("Video")
            self.videoLoop()
        elif self.isImage:
            print("Image")
            self.imageLoop()

    def get_shape(self, cnt):
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True) # calculates a contour perimeter or a curve length
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True) # approximates the polygonal curve with the specified precision

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            ar = w / float(h) # aspect ratio
            if ar >= 0.95 and ar <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
        else:
            shape = "circle"

        return shape

    def draw_centered_text(self, img, text, pos, color, font_scale, thickness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = pos[0] - size[0] // 2
        y = pos[1] + size[1] // 2
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


    def get_similarities(self, test_image, reference_image):
        sift = cv2.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(reference_image, None)
        kp_2, desc_2 = sift.detectAndCompute(test_image, None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    
        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio*n.distance: # if the distance is less than 60% of the second distance, then it's a good match
                good_points.append(m)
    
        frame_keypoints = cv2.drawMatches(test_image, kp_1, reference_image, kp_2, good_points, None)
        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        
        return (frame_keypoints,len(good_points) / number_keypoints * 100) # return the percentage of similarity and the image with the matches

    def get_pre_processed_similarities(self, test_image : np.ndarray, reference_image_pre_processed) -> float:
        sift = cv2.SIFT_create()

        t1 = time.time()
        kp_1 = Utils.deserialize(reference_image_pre_processed["kp"])
        desc_1 = reference_image_pre_processed["desc"]

        kp_2, desc_2 = sift.detectAndCompute(test_image, None)
    


        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    

        good_points = []
        ratio = 0.6
        for first_match, second_match in matches: 
            if first_match.distance < ratio*second_match.distance: # if the distance is less than 60% of the second distance, then it's a good match
                good_points.append(first_match)
                # print(len(good_points))

        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        

        return (len(good_points) / number_keypoints * 100)

    def get_best_match(self, test_image : np.ndarray, reference_image_pre_processed_dict : dict) -> dict:
        try:
            best_match = None
            best_match_similarity = 0
            for reference_image_path in reference_image_pre_processed_dict.keys():
                reference_image_pre_processed = reference_image_pre_processed_dict[reference_image_path]

                similarity = self.get_pre_processed_similarities(test_image, reference_image_pre_processed)
                similarities_img = None

                if similarity > best_match_similarity:
                    best_match_similarity = similarity
                    best_match_path = reference_image_path
                    best_match = cv2.imread(reference_image_path)

            if best_match is None:
                return None
            else:
                return {"path": best_match_path, "image": best_match, "similarities": best_match_similarity, "similarities_img": similarities_img}
        except Exception as e:
            cprint("Error in get_best_match: ", "red", attrs=["bold"], end="")
            #print(e)
            raise e
            return None


    def get_detection_best_match(self, best_matches):
        # return the element that appears the most in the best_matches list
        counts = {}
        for best_match in best_matches:
            counts[best_match] = counts.get(best_match, 0) + 1 # counts.get(best_match, 0) gets the value of the key, if it doesn't exist, return 0
            
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True) # sort the dictionary by the value

        return sorted_counts[0][0]

    def format_cnt(self, cnt):
        formatted_cnt = []
        for i in range(len(cnt)):
            formatted_cnt.append(cnt[i][0])

        return np.array(formatted_cnt)

    def imageLoop(self):

        originale_frame = cv2.imread(self.file_path)

        has_detected = False
        is_detecting = False
        best_matches = []
        overall_best_match = None

        frame = originale_frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        outputs = []
        masks = []
        for (lower, upper) in self.boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # find the colors within the specified boundaries and apply the mask
            mask = cv2.inRange(hsv, lower, upper)
            # mask = cv2.dilate(mask, None, iterations=2)         
            masks.append(mask)

            output = cv2.bitwise_and(originale_frame, originale_frame, mask = mask)
            outputs.append(output)

        # combine the outputs
        output = np.zeros_like(outputs[0])
        mask = np.zeros_like(masks[0])
        for circle in range(len(outputs)):
            output = cv2.add(output, outputs[circle])
            mask = cv2.add(mask, masks[circle])

        contours  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        frame = cv2.drawContours(frame, contours, -1, (255, 255, 0), 1) # cyan
            
        circles_cnt = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100 :
                # circles_cnt.append(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 15 and h > 15:

                    # print("w: " + str(w) + " h: " + str(h))
                    if w / h > 0.8 and w / h < 1.2: # if the width and height are similar, it's a circle, or a square, or a triangle
                        if self.get_shape(cnt) == "circle":
                            # print(self.get_shape(cnt))
                            circles_cnt.append(cnt)

        frame = cv2.drawContours(frame, circles_cnt, -1, (255, 0, 255), 1) # color is magenta

        # print("Number of circles: " + str(len(circles_cnt)))
        
        # sorted_cnt = sorted(circles_cnt, key=cv2.contourArea, reverse=True)
        # contours = self.format_cnt(circles_cnt)

        # only keep the contours that aren't contained in another contour using the hierarchy
        filtered_circles_cnt = []
        for i, cnt in enumerate(circles_cnt):
            try:
                is_contained = False
                to_test = circles_cnt.copy()
                del to_test[i]

                cnt = self.format_cnt(cnt)
                for cnt2 in to_test:
                    if cnt2 is not cnt:
                        cnt2 = self.format_cnt(cnt2)

                        if cv2.pointPolygonTest(cnt2, tuple(cnt[0].tolist()), False) == 1:
                            is_contained = True
                            break
                if not is_contained:
                    filtered_circles_cnt.append(cnt)
            except Exception as e:
                cprint("Error while filtering contours", "red", attrs=["bold"])
                print(e)
                
        frame = cv2.drawContours(frame, filtered_circles_cnt, -1, (0, 255, 0), 1) # color is green
        
        # if len(filtered_circles_cnt) > 0:
        #     for cnt in filtered_circles_cnt:
        #         x, y, w, h = cv2.boundingRect(cnt)
                # cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)


        extracted_images = []
        rects = []
        if len(filtered_circles_cnt) > 0:
            is_detecting = True
            has_detected = True
            i = 0
            for cnt in filtered_circles_cnt:
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append((x, y, w, h))

                extracted_image = originale_frame[y:y+h, x:x+w]
                extracted_images.append(extracted_image)

                # cv2.im
                (f"extracted_image", extracted_image)
                i += 1
        else:
            # print("No circles found")
            is_detecting = False
        
        # cv2.imshow(f"images", np.hstack([frame, output]))
    
        similarities_img = []
        result = originale_frame.copy()
        for i, image in enumerate(extracted_images):
            # cv2.imshow("image" + str(i), image)
            # image = wraped_images[i]
            best_match_dict = self.get_best_match(image, self.reference_image_pre_processed_dict)

            if not best_match_dict is None:
                best_matches.append(best_match_dict["path"]) # add the best match to the list of best matches

                best_match_frame = best_match_dict["image"]
                # cv2.imshow("similarities" + str(i), self.get_similarities_img(image, best_match_frame))

                # concatenate the best match with the original image (they are not the same size)
                best_match_frame = cv2.resize(best_match_frame, (image.shape[1], image.shape[0]))

                x, y, w, h = rects[i]
                match_x = x - (best_match_frame.shape[1] - w)
                match_y = y - (best_match_frame.shape[0] - h)
                result[match_y:match_y+best_match_frame.shape[0], match_x:match_x+best_match_frame.shape[1]] = best_match_frame
            else:
                print("No match found for image " + str(i))
        
        if has_detected and not is_detecting:
            if len(best_matches) > 0:
                overall_best_match = self.get_detection_best_match(best_matches)
                best_matches = [] # reset the best matches if we are not detecting

            if not overall_best_match is None:
                overall_best_match_img = cv2.imread(overall_best_match)
                
                # put the overall best match in the top left corner
                scale = 50/100
                resized_shape = (int(overall_best_match_img.shape[1]*scale), int(overall_best_match_img.shape[0]*scale))
                overall_best_match_img = cv2.resize(overall_best_match_img, (resized_shape[0], resized_shape[1]))
                result[:overall_best_match_img.shape[0], :overall_best_match_img.shape[1]] = overall_best_match_img

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(result)
        image_result_tk = ImageTk.PhotoImage(image)
        self.canvas.config(width=image_result_tk.width(), height=image_result_tk.height())
        self.canvas.create_image(0, 0, anchor=NW, image=image_result_tk)
        self.canvas.image = image_result_tk

        self.launch_button.pack_forget()

        self.root.update()

    def videoLoop(self):
        IS_PAUSED = False

        video = cv2.VideoCapture(self.file_path)

        has_detected = False
        is_detecting = False
        number_of_frame_in_detection = 0
        best_matches = []
        overall_best_match = None

        START_FRAME = 0
        video.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

        fps_start_time = time.time()
        fps = 0
        frame_count = 0

        average_time_to_find_match = 0
        nb_times = 0
        
        old_sign="" #sert pour ne pas repeter l'audio à chaque frame
        
        while True:
            frame_count += 1

            frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
            ret, originale_frame = video.read()

            if not ret:
                break

            frame = originale_frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

            outputs = []
            masks = []
            for (lower, upper) in self.boundaries:
                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")

                # find the colors within the specified boundaries and apply the mask
                mask = cv2.inRange(hsv, lower, upper)
            
                masks.append(mask)

                output = cv2.bitwise_and(originale_frame, originale_frame, mask = mask)
                outputs.append(output)

            # combine the outputs
            output = np.zeros_like(outputs[0])
            mask = np.zeros_like(masks[0])
            for circle in range(len(outputs)):
                output = cv2.add(output, outputs[circle])
                mask = cv2.add(mask, masks[circle])

            contours  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            frame = cv2.drawContours(frame, contours, -1, (255, 255, 0), 1)
                
            circles_cnt = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 15 and h > 15:

                
                    if w / h > 0.8 and w / h < 1.2: # if the width and height are similar, it's a circle, or a square
                        peri = cv2.arcLength(cnt, True) # get the perimeter of the contour

                        # check if the permimeter is similar to the perimeter of a circle
                        if peri / (2 * np.pi * (w / 2)) > 0.8 and peri / (2 * np.pi * (w / 2)) < 1.2:
                            circles_cnt.append(cnt)

            frame = cv2.drawContours(frame, circles_cnt, -1, (255, 0, 255), 1) # draw the contours in magenta


            # only keep the contours that aren't contained in another contour
            filtered_circles_cnt = []
            for i, cnt in enumerate(circles_cnt):
                is_contained = False
                to_test = circles_cnt.copy()
                del to_test[i]

                cnt = self.format_cnt(cnt)
                for test_cnt in to_test:
                    if test_cnt is not cnt:
                        test_cnt = self.format_cnt(test_cnt)

                        if cv2.pointPolygonTest(test_cnt, tuple(cnt[0].tolist()), False) == 1: # if the first point of the contour is inside the test contour
                            is_contained = True
                            break
                if not is_contained:
                    filtered_circles_cnt.append(cnt)
                    
            frame = cv2.drawContours(frame, filtered_circles_cnt, -1, (0, 255, 0), 1) # draw the filtered contours in green
            
            # ---- draw the bounding boxes around the circles ----
            # if len(filtered_circles_cnt) > 0:
            #     for cnt in filtered_circles_cnt:
            #         x, y, w, h = cv2.boundingRect(cnt)
            #         cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)
            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #         cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

            extracted_images = []
            rects = []
            if len(filtered_circles_cnt) > 0:
                
                is_detecting = True
                print("detected")
                number_of_frame_in_detection += 1
                has_detected = True
                i = 0
                for cnt in filtered_circles_cnt:
                    x, y, w, h = cv2.boundingRect(cnt)
                    rects.append((x, y, w, h))

                    extracted_image = originale_frame[y:y+h, x:x+w]
                    extracted_images.append(extracted_image)

                    #cv2.imshow(f"extracted_image", extracted_image)
                    i += 1
            else:
                # print("No circles found")
                is_detecting = False
            
                print("not detected")
            similarities_img = []
            result = originale_frame.copy()
            for i, image in enumerate(extracted_images):
                t1 = time.time()
                best_match_dict = self.get_best_match(image, self.reference_image_pre_processed_dict)
                
                #interface demonstration
                if self.demo:
                    if best_match_dict is not None:
                        if image is not None and best_match_dict["image"] is not None:
                            target_size = (500, 500)  # (largeur, hauteur)
                
                            # Redimensionner les deux images à la taille cible
                            resized_image1 = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                            resized_image2 = cv2.resize(best_match_dict["image"], target_size, interpolation=cv2.INTER_AREA)
                            result1 = self.get_similarities(resized_image1, resized_image2)
                            if result is not None:
                                a, b = result1
                                resized_image3 = cv2.resize(result, target_size, interpolation=cv2.INTER_AREA)
                                # Ajouter du texte à l'image 'a'
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                text = f"pourcentage de bon points : {b}"
                                position = (50, 50)  # Position (x, y) du texte sur l'image
                                font_scale = 1  # Taille de la police
                                color = (255, 0, 0)  # Couleur du texte (B, G, R)
                                thickness = 2  # Épaisseur du trait
                                cv2.putText(a, text, position, font, font_scale, color, thickness)
                                # Afficher les images avec le texte
                                cv2.imshow("point caracteristiques best match", np.hstack([a, resized_image3]))
                
                if nb_times == 0:
                    average_time_to_find_match = time.time() - t1
                else:
                    average_time_to_find_match = (average_time_to_find_match * nb_times + time.time() - t1) / (nb_times + 1)
                
                if not best_match_dict is None:
                    best_matches.append(best_match_dict["path"]) # add the best match to the list of best matches
            
                    best_match_frame = best_match_dict["image"]
            
                    # if best_match_dict["similarities_img"] is not None:
                    #     cv2.imshow("similarities " + str(i), best_match_dict["similarities_img"])
            
                    # concatenate the best match with the original image
                    best_match_frame = cv2.resize(best_match_frame, (image.shape[1], image.shape[0]))
                    x, y, w, h = rects[i]
                    match_x = x - (best_match_frame.shape[1] - w)
                    match_y = y - (best_match_frame.shape[0] - h)
                    result[match_y:match_y+best_match_frame.shape[0], match_x:match_x+best_match_frame.shape[1]] = best_match_frame
                else:
                    # print("No match found for image " + str(i))
                    pass
            overall_best_match_img= cv2.imread("rientrouve.jpg")
            if has_detected and not is_detecting:
                if number_of_frame_in_detection > 5:
                    if len(best_matches) > 0:
                        filepaths = best_matches
                        # Compter les occurrences de chaque fichier dans la liste
                        file_count = Counter(filepaths)
                        
                        # Trier les fichiers par nombre d'occurrences décroissant
                        sorted_file_count = sorted(file_count.items(), key=lambda x: x[1], reverse=True)
                        
                        print("Panneaux detectés:")
                        for file, count in sorted_file_count:
                            print(f"{file}: {count} fois")
                        overall_best_match = self.get_detection_best_match(best_matches)
                        # reset the best matches and the number of frame in detection for the next detection
                        best_matches = []
                        number_of_frame_in_detection = 0

                
                if not overall_best_match is None:
                    overall_best_match_img = cv2.imread(overall_best_match)
                    
                    
                    # put the overall best match in the top left corner
                    scale = 50/100
                    resized_shape = (int(overall_best_match_img.shape[1]*scale), int(overall_best_match_img.shape[0]*scale))
                    overall_best_match_img = cv2.resize(overall_best_match_img, (resized_shape[0], resized_shape[1]))
                    result[:overall_best_match_img.shape[0], :overall_best_match_img.shape[1]] = overall_best_match_img
                    
                    current_sign=overall_best_match
                    #audio lisant le panneau détecté
                    if current_sign != old_sign :
                        if os.path.basename(overall_best_match) == "ref30.jpg":
                            self.label.config(text="limitation de vitesse à 30")
                            self.engine.say("limitation de vitesse à 30")
                            self.engine.runAndWait()
                            old_sign=current_sign
                        elif os.path.basename(overall_best_match) == "ref50.jpg":
                            self.label.config(text="limitation de vitesse à 50")
                            self.engine.say("limitation de vitesse à 50")
                            self.engine.runAndWait()
                            old_sign=current_sign
                        elif os.path.basename(overall_best_match) == "ref70.jpg":
                            self.label.config(text="limitation de vitesse à 70")
                            self.engine.say("limitation de vitesse à 70")
                            self.engine.runAndWait()
                            old_sign=current_sign    
                        elif os.path.basename(overall_best_match) == "ref90.jpg":
                            self.label.config(text="limitation de vitesse à 90")
                            self.engine.say("limitation de vitesse à 90")
                            self.engine.runAndWait()
                            old_sign=current_sign
                        elif os.path.basename(overall_best_match) == "ref110.jpg":
                            self.label.config(text="limitation de vitesse à 110")
                            self.engine.say("limitation de vitesse à 110")
                            self.engine.runAndWait()
                            old_sign=current_sign
                        elif os.path.basename(overall_best_match) == "refdouble.jpg":
                            self.label.config(text="Interdiction de doubler")
                            self.engine.say("Interdiction de doubler")
                            self.engine.runAndWait()
                            old_sign=current_sign
                        else :
                            self.label.config(text="  ")

            # add average time to find a match to the image
            frame = cv2.putText(frame, f"average time to find a match: {average_time_to_find_match}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # add the frame number to the image
            # frame = cv2.putText(frame, f"frame number: {frame_number}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if frame_count == 10:
                fps = int(frame_count / (time.time() - fps_start_time))
                frame_count = 0
                fps_start_time = time.time()

            # add the fps to the image
            frame = cv2.putText(frame, f"fps: {fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Redimensionnement d'image
            height = originale_frame.shape[0]
            width = int(overall_best_match_img.shape[1] * (height / overall_best_match_img.shape[0]))
            resized_overall_best_match_img = cv2.resize(overall_best_match_img, (width, height), interpolation=cv2.INTER_AREA)

            
            
            #Interface client
            cv2.imshow("frame", frame)
            
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(result)
            image_result_tk = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=NW, image=image_result_tk)
            self.root.update()
            self.canvas.delete("all")
            
            
            if not IS_PAUSED:
                delay = int(1000 / video.get(cv2.CAP_PROP_FPS))
                key = cv2.waitKeyEx(delay)
            else:
                key = cv2.waitKeyEx(0)
                # print(f"key: {key}")

            # pause the video if the space bar is pressed
            if key == ord(" "):
                IS_PAUSED = not IS_PAUSED

            # if the key is e, go to the next frame
            if key == ord("e"):
                continue
                
            # if the key is a, go to the previous frame
            if key == ord("a"):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            
            if key == ord("q"):
                video.release()
                cv2.destroyAllWindows()
                break

        video.release()
        cv2.destroyAllWindows()

        self.canvas.delete("all")
        self.canvas.config(width=0, height=0)

        self.canvas.pack_forget()
        self.launch_button.pack_forget()
        
        
        self.root.update()
        
        

    def run(self):
        self.root.mainloop()


interface = Interface(demo=False)
interface.run()