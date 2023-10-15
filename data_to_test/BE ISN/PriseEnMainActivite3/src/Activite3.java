
import org.opencv.core.*;
import org.opencv.highgui.*;

public class Activite3 {

	public static void main(String[] args)
	{
		System.loadLibrary("opencv_java249");
		Mat m=Highgui.imread("p0.jpg",Highgui.CV_LOAD_IMAGE_COLOR);
		MaBibliothequeTraitementImage.afficheImage("Image originale", m);
				
	
	}
}