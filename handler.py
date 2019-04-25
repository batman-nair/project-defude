from gi.repository import Gtk, GdkPixbuf
import gtk
import threading
import os
from defocus.defocus import DefocuserObject

class Wait(object):
    def __init__(self,handler,status_line):
        self.handler = handler
        self.status_line = status_line
        self.prev_status_line = self.handler.status_line.get_text()

    def __enter__(self):
        print("entering context",self.status_line)
        self.handler.status_spinner.start()
        self.handler.status_line.set_text(self.status_line)
    
    def __exit__(self, *args):
        print("exiting context")
        self.handler.status_spinner.stop()
        self.handler.status_line.set_text(self.prev_status_line)


class Handler(object):
    def __init__(self,builder,checkpoint_path):
        self.original_image = builder.get_object('orignal_image')
        self.processed_image = builder.get_object('processed_image')
        self.status_line = builder.get_object('status_line')
        self.status_spinner = builder.get_object('status_spinner')
        self.result_type_picker = builder.get_object('result_type_picker')

        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH  = 512

        self.checkpoint_path = checkpoint_path
        self.depthmap_path = None
        self.input_image_path = None
        self.DISPLAY_TEXT = {
            'depth_est_running': 'Estimating Depthmap...',
            'idle': 'Done!',
            'defocus_running': 'Performing defocussing...'
        }

        self.result_type_picker.set_sensitive(False)

    
    def onImageClick(self,*args):
        event = args[1]
        x,y = event.x, event.y
        x_norm = x / self.IMAGE_WIDTH
        y_norm = y / self.IMAGE_HEIGHT

        if self.input_image_path is not None:
            thread = threading.Thread(target=lambda: self._defocuss_image(x_norm,y_norm))
            thread.daemon = True
            thread.start()

    def _defocuss_image(self,x_norm,y_norm):
        self.status_line.set_text(self.DISPLAY_TEXT['defocus_running'])
        self.status_spinner.start()
        defocusser = DefocuserObject(self.input_image_path)
        defocusser.set_pof_from_coord(x_norm,y_norm)
        self.status_line.set_text(self.DISPLAY_TEXT['idle'])
        self.status_spinner.stop()
        self.result_type_picker.set_sensitive(True)
        self.result_type_picker.set_active_id('defocussed_image')
        filename_no_ext = os.path.join(os.path.dirname(self.input_image_path), os.path.basename(self.input_image_path).split('.')[0])
        pixbuf = self._resize_image(filename_no_ext + '_defocus.png')
        self.processed_image.set_from_pixbuf(pixbuf)


    def _generate_depthmap(self,filename):
        print("starting")
        self.result_type_picker.set_sensitive(False)
        self.result_type_picker.set_active_id('depth_map')
        self.status_line.set_text(self.DISPLAY_TEXT['depth_est_running'])
        self.status_spinner.start()
        os.system("python ./depth/depth_simple.py --checkpoint_path " + self.checkpoint_path + " --image_path " + filename)        
        self.status_spinner.stop()
        self.status_line.set_text(self.DISPLAY_TEXT['idle'])
        filename_no_ext = os.path.join(os.path.dirname(filename), os.path.basename(filename).split('.')[0])
        print(filename_no_ext + '_disp.png')
        pixbuf = self._resize_image(filename_no_ext + '_disp.png')
        self.processed_image.set_from_pixbuf(pixbuf)
        print("end")



    def onDestroy(self, *args):
        Gtk.main_quit()

    def onResultPickerChange(self,picker):
        active_id = picker.get_active_id()
        filename_no_ext = os.path.join(os.path.dirname(self.input_image_path), os.path.basename(self.input_image_path).split('.')[0])
        pixbuf = None
        if active_id == 'depth_map':
            pixbuf = self._resize_image(filename_no_ext+'_disp.png')
        else:
            pixbuf = self._resize_image(filename_no_ext+'_defocus.png')
        
        self.processed_image.set_from_pixbuf(pixbuf)


    def onInputImageFileSet(self, *args):
        input_file_name = args[0].get_filename()
        pixbuf = self._resize_image(input_file_name)
        self.original_image.set_from_pixbuf(pixbuf)
        self.processed_image.set_from_pixbuf(pixbuf)
        self.input_image_path = input_file_name
        thread = threading.Thread(target=lambda: self._generate_depthmap(input_file_name))
        thread.daemon = True
        thread.start()
        
    def _resize_image(self,filename):
        pixbuf =  GdkPixbuf.Pixbuf.new_from_file(filename)
        pixbuf = pixbuf.scale_simple(self.IMAGE_WIDTH,self.IMAGE_HEIGHT,GdkPixbuf.InterpType.BILINEAR)
        return pixbuf