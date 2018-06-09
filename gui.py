
def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    size = '%dx%d+%d+%d' %(width,height,(screen_width-width)/2, (screen_height-height)/2)
    root.geometry(size)