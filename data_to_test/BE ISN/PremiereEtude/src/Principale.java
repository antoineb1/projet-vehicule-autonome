
import org.opencv.core.*;
import org.opencv.highgui.*;

public class Principale {

	public static void main(String[] args)
	{
		System.loadLibrary("opencv_java249");

		Mat imageOriginale=Highgui.imread("Temoin.png",Highgui.CV_LOAD_IMAGE_COLOR);
		Mat imageTransformee=MaBibliothequeTraitementImage.transformeBGRversHSV(imageOriginale);
		Mat imageSatureExemple=MaBibliothequeTraitementImage.seuillage_exemple(imageTransformee, 170);	
		//Mat imageSaturee=MaBibliothequeTraitementImage.seuillage(imageTransformee, 6, 170, 110);
				
		MaBibliothequeTraitementImage.afficheImage("Image originale", imageOriginale);
		MaBibliothequeTraitementImage.afficheImage("Saturation des rouges", imageSatureExemple);
	
	}
}