import java.io.File;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class Activite1 {

	public static void main(String[] args) {
		//Chargement de la librairie
		System.loadLibrary("opencv_java249");
		//Lecture du fichier image et transformation en matrice
		Mat m=LectureImage("activite1.png");
		
		//Début du traitement
		for (int i=0; i<m.height();i++){
			for(int j=0;j<m.width();j++){
				double[] BGR=m.get(i, j);
				//si c'est un point blanc....
				if(BGR[0]==255 && BGR[1]==255 && BGR[2]==255){
					//affiche un point dans la console sans retour à la ligne
					System.out.print(".");}
				else{//sinon...
					//affiche un + dans la console 
					System.out.print("+");
				}
			}
			//N'affiche rien mais execute un retour à la ligne
			System.out.println();
		}

	}
	//Methode qui ouvre un fichier et renvoie la matrice au format OpenCV
	public static Mat LectureImage(String fichier){
		File f=new File(fichier);
		Mat m=Highgui.imread(f.getAbsolutePath());
		return m;

	}


}
