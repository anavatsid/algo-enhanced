import cv2
import numpy as np

COLORS = np.random.uniform(0, 255, size=(2, 3))


class LS_Detector:
    def __init__(self, weights='model/detects_final.weights', config='model/config', label_file='model/names'):
        self.weights = weights
        self.config = config
        self.label_file = label_file
        self.net = None
        self.labels = None
        self.load_model()

    def load_model(self):
        # read class names from text file
        with open(self.label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNet(self.weights, self.config)

    # function to get the output layer names
    # in the architecture
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()

        try:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.labels[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def infer(self, image_data):

        res_data = {
            "success": False,
            "descript": "",
            "frame": None,
            "signal": None
        }

        if isinstance(image_data, str):
            img = cv2.imread(image_data)
        elif type(image_data).__module__ == np.__name__:
            img = image_data
        else:
            res_data["descript"] = "Invalid Input type..."
            return res_data
        
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # scale = 1
        Width = img.shape[1]
        Height = img.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        output_layers = self.get_output_layers()
        outs = self.net.forward(output_layers)

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.3
        nms_threshold = 0.3

        center_x_points = []

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                # print(f"{detection=}")
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    # if center_x
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    center_x_points.append(center_x)
        if len(boxes) != 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold).flatten().tolist()
            colors = [(255, 0, 0), (0, 255, 0), (128, 128, 0)]
            last_color = (0, 0, 255)
            nms_center_x_points = [center_x_points[i] for i in indices]
            right_most_id = center_x_points.index(max(nms_center_x_points))

            final_label = self.labels[class_ids[right_most_id]]

            for index in indices:
                # index = index[0]
                x, y, w, h = boxes[index]
                # cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[index]], 2)
                # cv2.putText(img, self.labels[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #             colors[class_ids[index]], 2)

            x, y, w, h = boxes[right_most_id]
            cv2.rectangle(img, (x, y), (x + w, y + h), last_color, 2)
            cv2.putText(img, final_label, (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        last_color, 2)
        else:
            final_label = None

        res_data["success"] = True
        res_data["descript"] = "Successfully Processed"
        res_data["frame"] = img
        res_data["signal"] = final_label

        return res_data

if __name__ == "__main__":
    image_path = 'E:\\work_kss\\Yolo_train\\darknet\\build\darknet\\x64\\TOTAL\\images\\NQ1_009.jpg'
    processor = LS_Detector(weights='model/yolov3-tiny-custom_final.weights')
    result = processor.infer(image_path)
    # img = cv2.resize(img, (Width*2,Height*2))
    cv2.imshow("image", result["frame"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
