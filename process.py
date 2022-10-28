# from cProfile import label
from ast import Delete
import cv2
from skimage.metrics import structural_similarity
import numpy as np

COLORS = np.random.uniform(0, 255, size=(2, 3))


class LSC_Detector:
    def __init__(self, weights='model/detects_final.weights', config='model/config', label_file='model/names'):
        self.weights = weights
        self.config = config
        self.label_file = label_file
        self.net = None
        self.labels = None
        self._prev_ = {}
        self.temp_signals = []
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
    def label_tracking(self):
        if len(self.temp_signals) < 3:
            return False
        first_signal = self.temp_signals[0]
        for signal in self.temp_signals[1:]:
            if signal != first_signal:
                return False
        return True

    def tracking(self, final_label, box, img):
        if self.label_tracking():
            pass
        else:
            return self._prev_["label"]
        height, width = img.shape[:2]
        cur_x, cur_y, cur_w, cur_h = box
        # print(cur_x, cur_y, cur_w, cur_h)

        prev_x, prev_y, prev_w, prev_h = self._prev_["box"]
        w = max(cur_w, prev_w)
        w_gap = int(w/4)
        h = max(cur_h, prev_h)
        h_gap = int(h/4)

        if final_label != self._prev_["label"]:
            cur_x_left = max(0, cur_x - w_gap)
            cur_x_right = min(width, cur_x + w + w_gap)
            if cur_x_left == 0:
                cur_x_right = w + w_gap * 2
            elif cur_x_right == width:
                cur_x_left = int(width - (w + w_gap * 2))

            cur_y_top = max(0, cur_y - h_gap)
            cur_y_bottom = min(height, cur_y + h + h_gap)
            if cur_y_top == 0:
                cur_y_bottom = h + h_gap * 2
            elif cur_y_bottom == height:
                cur_y_top = int(height - (h + h_gap * 2))
            cur_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cur_cropped_box = cur_gray[cur_y_top:cur_y_bottom, cur_x_left:cur_x_right]
            # print(f"{cur_cropped_box.shape=}")

            prev_x_left = max(0, prev_x - w_gap)
            prev_x_right = min(width, prev_x + w + w_gap)
            if prev_x_left == 0:
                prev_x_right = w + w_gap * 2
            elif prev_x_right == width:
                prev_x_left = int(width - (w + w_gap * 2))

            prev_y_top = max(0, prev_y - h_gap)
            prev_y_bottom = min(height, prev_y + h + h_gap)
            if prev_y_top == 0:
                prev_y_bottom = h + h_gap * 2
            elif prev_y_bottom == height:
                prev_y_top = int(height - (h + h_gap * 2))

            prev_img = self._prev_["frame"]
            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            prev_cropped_box = prev_gray[prev_y_top:prev_y_bottom, prev_x_left:prev_x_right]

            # print(f"{prev_cropped_box.shape=}")

            # Compute SSIM between two images
            score, diff = structural_similarity(cur_cropped_box, prev_cropped_box, full=True)
            # print(f"\n{final_label=}\t{self._prev_['label']=}\t{score=}")
            if score <= 0.4:
                self._prev_["label"] = final_label
                self._prev_["frame"] = img
                self._prev_["box"] = (cur_x, cur_y, cur_w, cur_h)
            # else:
            #     self._prev_["box"] = (prev_x, prev_y, prev_w, prev_h)
        return self._prev_["label"]

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

        origin_img = img.copy()
        # print(origin_img.shape)
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
        conf_threshold = 0.5
        nms_threshold = 0.5

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
            # colors = [(255, 0, 0), (0, 255, 0), (128, 128, 0)]
            last_color = (0, 0, 255)
            nms_center_x_points = [center_x_points[i] for i in indices]
            right_most_id = center_x_points.index(max(nms_center_x_points))

            final_label = self.labels[class_ids[right_most_id]]
            if len(self.temp_signals) >= 3:
                del self.temp_signals[0]
            self.temp_signals.append(final_label)

            if self._prev_ != {}:
                # if self._prev_["label"] != final_label:
                final_label = self.tracking(final_label, boxes[right_most_id], img)
            else:
                x, y, w, h = boxes[right_most_id]
                self._prev_["frame"] = img
                self._prev_["box"] = (x, y, w, h)
                self._prev_["label"] = final_label
            # for index in indices:
            #     # index = index[0]
            #     x, y, w, h = boxes[index]
            #     cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[index]], 2)
            #     cv2.putText(img, self.labels[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #                 colors[class_ids[index]], 2)

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
    processor = LSC_Detector()
    result = processor.infer(image_path)
    # img = cv2.resize(img, (Width*2,Height*2))
    cv2.imshow("image", result["frame"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
