import gi

gi.require_version('Gtk','3.0')

from gi.repository import Gtk, Gdk, GdkPixbuf
import threading
from functools import partial
import os
from defocus.defocus import DefocuserObject
import cv2

class DefudeGui(object):

    def __init__(self,checkpoint_path,glade_file='ui-stepper.glade'):
        self.builder = Gtk.Builder()

        # load the glade file describing the UI
        self.builder.add_from_file(glade_file)

        # connect the event handlers
        self.builder.connect_signals(self)

        # all the available windows
        self.main_window = self.builder.get_object('main-window')
        self.about_dialog = self.builder.get_object('about-dialog')
        self.help_dialog = self.builder.get_object('help-dialog')


        # some ui elements
        self.input_image_drop = self.builder.get_object('input-image-drop')
        self.select_input_image_picker = self.builder.get_object('select-image-file-picker')
        self.step_stack = self.builder.get_object('main-step-stack')
        self.image_drop_dest = self.builder.get_object('image-drag-drop-dest')
        self.input_image_spinner = self.builder.get_object('input-image-spinner')
        self.input_image_status_line = self.builder.get_object('input-image-status')
        self.input_image_delete_btn = self.builder.get_object('delete-input-img-btn')
        self.input_image_next_btn = self.builder.get_object('input-image-next-btn')
        self.depth_map_image = self.builder.get_object('depth-map-image')
        self.pof_image = self.builder.get_object('pick-pof-image')
        self.pof_status_line = self.builder.get_object('pof-status-line')
        self.pof_status_spinner = self.builder.get_object('pof-status-spinner')
        self.result_image = self.builder.get_object('result-image')
        self.header_bar = self.builder.get_object('main-header-bar')

        page_ids = (
            'start-page',
            'select-image-page',
            'depth-map-preview',
            'pick-pof',
            'result-page'
        )

        self.pages = tuple(
            self.builder.get_object(page_id)
            for page_id in page_ids
        )

        # some useful constants
        self.IMAGE_WIDTH = 600
        self.CHECKPOINT_PATH = checkpoint_path
        self.STATUS_MESSAGES = {
            'depth_est_running': 'Estimating depthmap...',
            'defocus_running': 'Performing defocusing...',
            'idle': '',
            'resizing': 'Loading the image...'
        }
        self.DEFAULT_INPUT_IMAGE = 'assets/drag-and-drop.png'
        self.WINDOW_TITLE = 'Synthetic Defocusing and Depth Estimation Tool'

        # state variables
        self.INPUT_IMAGE_PATH = None
        self.INPUT_IMAGE_SIZE = None
        self.DEPTH_MAP_PATH = None
        self.DEFOCUS_IMAGE_PATH = None
        self.CURRENT_STACK_PAGE = 0

        # set the image as a drag drop destination
        self.image_drop_dest.drag_dest_set(
            # do all the default stuff
            Gtk.DestDefaults.ALL,
            # enforce target
            [Gtk.TargetEntry.new('text/plain',Gtk.TargetFlags(4), 129)],
            Gdk.DragAction.COPY
        )

    def _next_page(self):
        if self.CURRENT_STACK_PAGE < len(self.pages) - 1:
            self.CURRENT_STACK_PAGE += 1
            self.step_stack.set_visible_child(self.pages[self.CURRENT_STACK_PAGE])

    def _prev_page(self):
        if self.CURRENT_STACK_PAGE > 0:
            self.CURRENT_STACK_PAGE -= 1
            self.step_stack.set_visible_child(self.pages[self.CURRENT_STACK_PAGE])

    def _set_input_image_impl(self, img_path):
        spinner = self.input_image_spinner
        status_line = self.input_image_status_line
        self.INPUT_IMAGE_PATH = img_path
        img,w,h = self._resize_image(img_path,return_size=True)
        self.INPUT_IMAGE_SIZE = (w,h)
        self.input_image_drop.set_from_pixbuf(img)
        self.select_input_image_picker.hide()

        spinner.stop()
        status_line.set_label(self.STATUS_MESSAGES['idle'])
        self.input_image_delete_btn.set_sensitive(True)
        self.input_image_next_btn.set_sensitive(True)

    def _set_input_image(self, img_path):

        spinner = self.input_image_spinner
        status_line = self.input_image_status_line
        spinner.start()
        status_line.set_label(self.STATUS_MESSAGES['resizing'])

        thread = threading.Thread(target=partial(self._set_input_image_impl,img_path))
        thread.daemon = True
        thread.start()


    def _unset_input_image(self):
        self.INPUT_IMAGE_PATH = None
        self.INPUT_IMAGE_SIZE = None
        self.input_image_drop.set_from_file(self.DEFAULT_INPUT_IMAGE)
        self.select_input_image_picker.show()
        self.input_image_next_btn.set_sensitive(False)

    def _resize_image(self,path,size=None,return_size=False):
        img = GdkPixbuf.Pixbuf.new_from_file(path)
        if size is None:
            h = img.get_height()
            w = img.get_width()

            ar = h / float(w)
            size = (self.IMAGE_WIDTH,int(self.IMAGE_WIDTH*ar))

        resized = img.scale_simple(size[0],size[1],GdkPixbuf.InterpType.BILINEAR)

        if return_size:
            return resized,size[0],size[1]

        return resized

    def _estimate_depthmap_impl(self):
        os.system("python ./depth/depth_simple.py --checkpoint_path " + self.CHECKPOINT_PATH + " --image_path " + self.INPUT_IMAGE_PATH)
        self.DEPTH_MAP_PATH = os.path.join(os.path.dirname(self.INPUT_IMAGE_PATH), os.path.basename(self.INPUT_IMAGE_PATH).split('.')[0]) + '_disp.png'

        depthmap_img = self._resize_image(self.DEPTH_MAP_PATH)
        self.depth_map_image.set_from_pixbuf(depthmap_img)

        self.input_image_spinner.stop()
        self.input_image_status_line.set_label(self.STATUS_MESSAGES['idle'])
        self._next_page()
        self.input_image_next_btn.set_sensitive(True)
        self.input_image_delete_btn.set_sensitive(True)



    def _estimate_depthmap(self):
        self.input_image_next_btn.set_sensitive(False)
        self.input_image_delete_btn.set_sensitive(False)
        self.input_image_spinner.start()
        self.input_image_status_line.set_label(self.STATUS_MESSAGES['depth_est_running'])

        thread = threading.Thread(target=self._estimate_depthmap_impl)
        thread.daemon = True
        thread.start()

    def _defocus_image_impl(self, x_norm, y_norm):
        defocusser = DefocuserObject(self.INPUT_IMAGE_PATH)
        defocusser.set_pof_from_coord(x_norm,y_norm)
        self.DEFOCUS_IMAGE_PATH = os.path.join(os.path.dirname(self.INPUT_IMAGE_PATH), os.path.basename(self.INPUT_IMAGE_PATH).split('.')[0]) + '_defocus.png'
        img = self._resize_image(self.DEFOCUS_IMAGE_PATH)
        self.result_image.set_from_pixbuf(img)
        self.pof_status_spinner.stop()
        self.pof_status_line.set_label(self.STATUS_MESSAGES['idle'])
        self._next_page()

    def _defocus_image(self, x_norm, y_norm):
        self.pof_status_spinner.start()
        self.pof_status_line.set_label(self.STATUS_MESSAGES['defocus_running'])
        thread = threading.Thread(target=partial(self._defocus_image_impl,x_norm,y_norm))
        thread.daemon = True
        thread.start()

    def _save(self, src_filename):
        dialog = Gtk.FileChooserDialog("Save as",self.main_window,Gtk.FileChooserAction.SAVE,(Gtk.STOCK_SAVE,Gtk.ResponseType.OK,Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL))
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            img = cv2.imread(src_filename)
            cv2.imwrite(dialog.get_filename(),img)
        dialog.destroy()

    def _cleanup(self):
        file_list = (
            self.DEFOCUS_IMAGE_PATH,
            self.DEPTH_MAP_PATH,
        )

        for file in file_list:
            if file:
                if os.path.isfile(file):
                    os.remove(file)


    def show(self):
        self.main_window.show_all()
        Gtk.main()

    def onDestroy(self, *args):
        Gtk.main_quit()
        # self._cleanup()

    def onStartPageNext(self, *args):
        self._next_page()

    def onBack(self, *args):
        self._prev_page()

    def onAbout(self, *args):
        self.about_dialog.run()
        self.about_dialog.hide()

    def onHelp(self, *args):
        self.help_dialog.run()
        self.help_dialog.hide()

    def onImagePickerSet(self, *args):
        input_file_name = args[0].get_filename()
        self._set_input_image(input_file_name)

    def onImageDrop(self, *args):
        filename = args[4].get_text()[7:-1]
        self._set_input_image(filename)

    def onDeleteInputImage(self, *args):
        self._unset_input_image()
        self.input_image_delete_btn.set_sensitive(False)

    def onInputImageNextBtn(self, *args):
        self._estimate_depthmap()

    def onDepthMapNextBtn(self, *args):
        img = self.input_image_drop.get_pixbuf()
        self.pof_image.set_from_pixbuf(img)
        self._next_page()

    def onDepthMapSave(self, *args):
        self._save(self.DEPTH_MAP_PATH)

    def onPofPick(self, *args):
        event = args[1]
        x,y = event.x, event.y
        x_norm = x / self.INPUT_IMAGE_SIZE[0]
        y_norm = y / self.INPUT_IMAGE_SIZE[1]
        print("x_norm: {} y_norm: {}".format(x_norm,y_norm))
        self._defocus_image(x_norm,y_norm)

    def onResultSave(self, *args):
        self._save(self.DEFOCUS_IMAGE_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic Defocussing Using Depth Estimation')

    parser.add_argument('--model_path', type=str, help='path to saved model', required=True)

    args = parser.parse_args()
    model_path = os.path.abspath(args.model_path)

    gui = DefudeGui(model_path)
    gui.show()
