import gi

gi.require_version('Gtk','3.0')

from gi.repository import Gtk
from handler import Handler


checkpoint_path = '/home/nm/Projects/project-defude/model'
builder = Gtk.Builder()
builder.add_from_file('ui.glade')

builder.connect_signals(Handler(builder,checkpoint_path))

win = builder.get_object('window1')

win.show_all()
Gtk.main()