import time
import cv2
import mss
import numpy


class ScreenCap:
   def __init__(self, box) -> None:
      left, top, width, height = box
      self.monitor = {'top': top, 'left': left, 'width': width, 'height': height}
      self.cap = None
      self.img = None
      self.fps = 5
      self.is_stop = False
      self.read()
      
   def isOpened(self):
      if self.img is not None:
         return True 
      else:
         return False

   def get(self, cap_prop_fps):
      return self.fps

   def set(self, cap_prop_fps, fps):
      self.fps = fps
   
   def read(self):
      with mss.mss() as sct:
         if not self.is_stop:
            self.img = numpy.array(sct.grab(self.monitor))
         else:
            self.img = None
      if self.img is not None:
         return True, self.img
      else:
         return False, None
   
   def release(self):
      self.is_stop = True

   def init(self):
      with mss.mss() as sct:
         while not self.is_stop:
            self.img = numpy.array(sct.grab(self.monitor))
                  # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
               cv2.destroyAllWindows()
               break
         self.img = None




      pass

