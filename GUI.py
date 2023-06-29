from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from painting_author import *
from sklearn.tree import DecisionTreeClassifier


image = None
data = []
train = []
test = []
featured_train = {}
train_size, seed, classes = get_size_and_seed()
methods = get_methods()
plt.rcParams["figure.figsize"] = [8, 5]
clf = None


class DesicionTree:
    def __init__(self, max_depth=10, min_samples_to_split=2):
        self.max_depth = max_depth
        self.min_sample_to_split = min_samples_to_split
        self.split_feature = None
        self.split_value = None
        self.left_child = None
        self.right_child = None
        self.label = None
    
    def gini(self, y):
        _, c = np.unique(y, return_counts=True)
        return 1 - np.sum((c / len(y)) ** 2)


    def fit(self, x, y):
        if not self.max_depth:
            self.max_depth = x.shape[1]
        
        if x.shape[0] < self.min_sample_to_split or self.max_depth == 0 or len(np.unique(y)) == 1:
            self.label = np.bincount(y).argmax()
            return
        
        best_feature = None
        best_value = None
        best_score = np.inf

        for f in range(x.shape[1]):
            for v in np.unique(x[:, f]):
                left = y[x[:, f] < v]
                right = y[x[:, f] >= v]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = (len(left) * self.gini(left) + len(right) * self.gini(right)) / len(y)
                if score < best_score:
                    best_score = score
                    best_feature = f
                    best_value = v
        
        self.split_feature = best_feature
        self.split_value = best_value

        self.left_child = DesicionTree(max_depth=self.max_depth-1, min_samples_to_split=self.min_sample_to_split)
        self.right_child = DesicionTree(max_depth=self.max_depth-1, min_samples_to_split=self.min_sample_to_split)
        self.left_child.fit(x[x[:, best_feature] < best_value], y[x[:, best_feature] < best_value])
        self.right_child.fit(x[x[:, best_feature] >= best_value], y[x[:, best_feature] >= best_value])

    def predict(self, x):
        if self.label is not None:
            return np.array([self.label] * x.shape[0])
        else:
            y = np.zeros(x.shape[0])
            y[x[:, self.split_feature] < self.split_value] = self.left_child.predict(x[x[:, self.split_feature] < self.split_value])
            y[x[:, self.split_feature] >= self.split_value] = self.right_child.predict(x[x[:, self.split_feature] >= self.split_value])
            return y.astype(int)


def browseFiles():
    if len(data) == 0:
        label_database.configure(text="You need to upload a database first!")
    else:
        global image
        image_path = filedialog.askopenfilename(initialdir="C:/biometrics/Paints/",
                                                title="Select a File",
                                                filetypes=[("Image files", "*.jpg *.png *.pgm")])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (350, 325))
        # cv2.imwrite("C:/biometrics/Paints/res.jpg", a)
        # plt.imshow(a), plt.show()
        label_database.configure(text="✓ File selected")


def load():
    label_database.configure(text="Database is uploading...")
    global data, train, test, featured_train, methods, clf
    data = load_paintings_from("./Paints/s", len(classes), 16)
    x_train, x_test, y_train, y_test = split_data_random(data, 16, train_size, seed=seed, SHOW=True)
    train = [x_train, y_train]
    test = [x_test, y_test]
    start = time.time()
    featured_train = {method.__name__: create_feature(x_train, method) for method in get_methods()}
    # clf = DesicionTree(max_depth=10, min_samples_to_split=2)
    clf = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_split=2)
    tree_train = np.zeros((len(x_train), len(methods)))
    i = 0
    j = 0
    for method in methods:
        features = create_feature(x_train, method)
        for feature in features:
            tree_train[i][j] = np.mean(feature)
            i += 1
        j += 1
        i = 0
    y = np.array(y_train)
    # clf.fit(tree_train, y)
    clf = clf.fit(tree_train, y)
    label_database.configure(text="✓ Database is uploaded")
    print(f"Database loaded in {int(time.time() - start)} seconds")


def accuracy():
    accuracy = test_voting(train, test, use_database=featured_train, clf=clf)
    print(f"Accuracy: {accuracy}")
    res = test_methods(train, test, clf=clf)
    print(f"Methods accuracy: {res}")


def classify():
    if image is None:
        label_database.configure(text="You need to upload an image first!")
    else:
        cols = (len(methods) + 1) // 3 + 2
        answer = voting(train, [[image], [0]], SHOW=True, use_database=featured_train, clf=clf, cols=cols)
        label_database.configure(text=classes[answer[0]])
        plt.subplot(3, cols, 3*cols, title="Result"), plt.axis("off")
        plt.text(0.3,
                 0.5,
                 classes[answer[0]],
                 transform=plt.gca().transAxes, fontdict={'size': 12})
        plt.show()


def show():
    if image is None:
        label_database.configure(text="You need to upload an image first!")
    else:
        draw_methods(image)
        plt.rcParams["figure.figsize"] = [8, 5]


def test():
    if len(data) == 0:
        label_database.configure(text="You need to upload a database first!")
    else:
        for test_img in test[0]:
            answer = voting(train, [[test_img], [0]], SHOW=True, use_database=featured_train)
            label_database.configure(text=classes[answer[0]])
            # plt.subplot(1, 2, 1, title="Query Image")
            # plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)), plt.axis("off")
            cols = len(methods) // 3 + 1
            plt.subplot(3, 3*cols, 3*cols, title="Result"), plt.axis("off")
            plt.text(0.3,
                     0.5,
                     classes[answer[0]],
                     transform=plt.gca().transAxes, fontdict={'size': 12})
            plt.show()
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def test_on_train():
    if len(data) == 0:
        label_database.configure(text="You need to upload a database first!")
    else:
        for train_img in train[0]:
            answer = voting(train, [[train_img], [0]], SHOW=True, use_database=featured_train)
            label_database.configure(text=classes[answer[0]])
            cols = len(methods) // 3 + 1
            plt.subplot(3, cols, 3*cols, title="Result"), plt.axis("off")
            plt.text(0.3,
                     0.5,
                     classes[answer[0]],
                     transform=plt.gca().transAxes, fontdict={'size': 12})
            plt.show()
            plt.show(block=False)
            plt.pause(2)
            plt.close()


if __name__ == "__main__":
    window = Tk()
    window.title('Paintings')
    window.geometry("640x480")
    window.config(background="white")

    bg = PhotoImage(file="./2.png")

    label1 = Label(window, image=bg)
    label1.place(x=150, y=10)
    label_database = Label(window)

    btn1 = Button(window, text="Load database", command=load)

    button_explore = Button(window, text="Select Image", command=browseFiles)

    btn2 = Button(window, text="Classify image", command=classify)

    btn3 = Button(window, text="Show methods", command=show)

    btn4 = Button(window, text="Accuracy", command=accuracy)

    # btn4 = Button(window, text="Test", command=test)

    # btn5 = Button(window, text="Test on train", command=test_on_train)

    label_database.grid(column=3, row=1)

    btn1.grid(column=1, row=1, pady=5, padx=10)
    button_explore.grid(column=1, row=2, pady=5)
    btn2.grid(column=1, row=4, pady=5)
    btn3.grid(column=1, row=3, pady=5)
    btn4.grid(column=1, row=5, pady=5)
    # btn5.grid(column=1, row=6, pady=5)

    window.mainloop()
