import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

public class Activite2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stubCE
		System.loadLibrary("opencv_java249");
		//Lecture du fichier image et transformation en matrice
		Mat m=LectureImage("bgr.png");
		//Création d'un vecteur de matrices
		Vector<Mat> channels=new Vector<Mat>();
		//découpage de la matrice en 3 canneaux séparés 
		Core.split(m,channels);
		for (int i=0;i<channels.size();i++){
			afficheImage("Voie n°"+Integer.toString(i),channels.get(i));
		}
	}


	//Methode qui ouvre un fichier et renvoie la matrice au format OpenCV (la meme que dans l'activite 0)
	public static Mat LectureImage(String fichier){
		File f=new File(fichier);
		Mat m=Highgui.imread(f.getAbsolutePath());
		return m;

	}
	//Methode qui affiche une matrice dans une fenetre Java (inutile d'analyser ce code dans le cadre de cette épreuve)
	public static void afficheImage(String title, Mat img){
		MatOfByte matOfByte=new MatOfByte();
		Highgui.imencode(".png",img,matOfByte);
		byte[] byteArray=matOfByte.toArray();
		BufferedImage bufImage=null;
		try{
			InputStream in=new ByteArrayInputStream(byteArray);
			bufImage=ImageIO.read(in);
			JFrame frame=new JFrame();
			frame.setTitle(title);
			frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
			frame.pack();
			frame.setVisible(true);

		}
		catch(Exception e){
			e.printStackTrace();
		}


	}

}
