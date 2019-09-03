from rest_framework.views import APIView
from rest_framework.response import Response
from faceapp.face_detector import predict,show_prediction_labels_on_image
import base64
class FaceRecognize(APIView):

    def post(self,request):
        img = request.data.get('image')
        print("In view",type(img))
        predictions = predict(img, model_path="trained_knn_model.clf")
        # draw_image = show_prediction_labels_on_image(img,predictions)
        names = []
        for name, (top, right, bottom, left) in predictions:
            names.append({"name":name})
        # names.append({'image':draw_image})
        return Response(names)

