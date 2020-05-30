import tkinter as tk

window = tk.Tk()

window.title('face_recognition')
window.geometry('500x300')

l = tk.Label(window, text='Hello!', bg='green', font=('Arial', 12), width=30, height=2)

l.pack()
window.mainloop()