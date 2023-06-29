import numpy as np
import random as rnd
import cv2
import matplotlib.pyplot as plt
from skimage.filters import sobel as sob
from skimage.filters import hessian, laplace, sato
import time
from skimage.measure import moments
from skimage.feature import hog
from mahotas import dog, euler, gaussian_filter, label, otsu
from mahotas.features import lbp, haralick, roundness
import imutils



# bl = cv2.medianBlur(img, ksize=15)
# fltr = cv2.bilateralFilter(img, 7, 100, 100)

def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 2:
        index, contour_arr = max(enumerate(contours[1:-1]), key=lambda x: len(x[1]))
        index += 1
    else: 
        index, contour_arr = 1, contours[1]
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, index, (255,255,255), 1)
    return contour_arr.squeeze(), contour_img


def get_start_point(contour_arr, mode="left-up"):
    mode_1, mode_2 = mode.split('-')
    def get_candidates(contour_arr, mode):
        if mode == "left":
            left_coord = contour_arr.min(0)[0]
            candidate_points = contour_arr[contour_arr[:, 0] == left_coord, :]
        elif mode == "right":
            right_coord = contour_arr.max(0)[0]
            candidate_points = contour_arr[contour_arr[:, 0] == right_coord, :]
        elif mode == "up":
            upper_coord = contour_arr.max(0)[1]
            candidate_points = contour_arr[contour_arr[:, 1] == upper_coord, :]
        elif mode == "down":
            down_coord = contour_arr.min(0)[1]
            candidate_points = contour_arr[contour_arr[:, 1] == down_coord, :]
        else:
            raise AttributeError
        return candidate_points
    
    candidates = get_candidates(contour_arr, mode_1)
    start_point = get_candidates(candidates, mode_2)
    return start_point.squeeze()


def get_vector_contour(contour_arr, start_point, contour_len=None):
    start_id = (contour_arr == start_point).sum(-1).argmax()
    contour_arr_start = np.concatenate([contour_arr[start_id:], contour_arr[:start_id]])
    if cv2.contourArea(contour_arr_start, True) < 0:
        contour_arr_start = np.concatenate([contour_arr_start[0:1, :], contour_arr_start[:0:-1, :]])
    deltas = contour_arr_start[1:] - contour_arr_start[:-1]
    contour_vector = [complex(x[0], x[1]) for x in deltas]
    last_delta = contour_arr_start[0] - contour_arr_start[-1]
    contour_vector.append(complex(last_delta[0], last_delta[-1]))
    contour_vector = np.array(contour_vector)
    if contour_len is not None:
        splt = np.array_split(contour_vector, contour_len)
        contour_vector = np.array([splt[i].sum() for i in range(len(splt))])
    return contour_vector


def NSP(c1, c2):
    if c1.shape != c2.shape:
        raise Exception(f"Sizes of contours are not the same: {c1.shape=}, {c2.shape=}")
    return np.vdot(c1, c2) / (np.sqrt(np.vdot(c1, c1))*np.sqrt(np.vdot(c2,c2)))


def ACF(c1, c2):
    if c1.shape != c2.shape:
        raise Exception(f"Sizes of contours are not the same: {c1.shape=}, {c2.shape=}")
    res = []
    for i in range(len(c2)):
        shifted2 = np.concatenate([c2[i:], c2[:i]])
        ac = np.vdot(c1, shifted2)
        res.append(ac)
    return np.array(res)


def similarity(c1, c2):
    return np.max(ACF(c1, c2) / (np.sqrt(np.vdot(c1, c1))*np.sqrt(np.vdot(c2,c2))))


def Moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    order = 3
    res = moments(gray, order=order)
    return res, res


def Euler(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T_otsu = otsu(gray)
    gray = gray > T_otsu
    eulr = np.abs(euler(gray))
    return eulr, eulr


def Haralick(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = gaussian_filter(img, 24)
    gaussian = (gaussian > gaussian.mean())
    labelled, n = label(gaussian)
    edges = haralick(labelled)
    return edges, edges


def LBP(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = lbp(img, 200, 10)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def Hessian(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = hessian(img)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = sob(img)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def Sato(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    edges = sato(gray)
    return edges, edges


def Laplace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    edges = laplace(gray)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def DOG(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    edges = dog(gray)
    return edges.astype(int), edges


def hough(img):
    res = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_size = (int(w * 0.5), int(h * 0.5))
    gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    circles_img = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=200, param2=50, minRadius=10, maxRadius=0)
    feature = 0
    if circles_img is not None:
        feature = circles_img.shape[1]
        circles_img = np.uint16(np.around(circles_img))
        for i in circles_img[0, :]:
            cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(res, (i[0], i[1]), 2, (0, 0, 255), 3)
    return feature, res


def harris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    res = np.copy(img)
    features = res[dst > 0.05 * dst.max()]
    res[dst > 0.05 * dst.max()] = [0, 0, 255]
    return features.shape[0], res


def HOG(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    return fd, hog_image


def gabor(img):
    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold
    filters = []
    ksize = 51
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=4.0, theta=theta, lambd=10.0,
                                  gamma=0.5, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum, accum


def fast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(50)
    kp = fast.detect(gray, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    return len(kp), img2


def canny(img):
    edges = cv2.Canny(img, 220, 230)
    amount_of_edges = len(edges[edges == 255])
    return amount_of_edges, edges


def color_hist(img):
    color = ('b', 'g', 'r')
    hists = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [64], [0, 256])
        hists.append(hist)
    return hists, hists


def random(img):
    np.random.seed(0)
    h, w, _ = img.shape
    features = []
    centers = []
    img2 = np.copy(img)
    for _ in range(400):
        x0 = int(np.random.rand() * w)
        y0 = int(np.random.rand() * h)
        centers.append((x0, y0))
        features.append(np.mean(img[y0, x0]))
        img2 = cv2.circle(img2, (x0, y0), 1, (0, 0, 255), 2)

    return features, img2


def DT_test(test, clf):
    methods = get_methods()
    answers = []
    for image in test:
        input_features = []
        for method in methods:
            input_features.append(np.mean(create_feature([image], method)))
        arr = np.array([input_features])
        # answers.append(clf.predict(arr)[0])
        answers.append(clf.predict([np.array(input_features)])[0])
    return answers


# def rectangs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(img, 10, 230)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rectangs = 0
    a = []
    for c in cnts:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*p, True)
        if len(approx) == 4:
            rectangs += 1
            a = cv2.drawContours(img.copy(), [approx], -1, (0, 255,), 4)
    return rectangs, a


def get_methods():
    return [color_hist, fast, random, LBP, canny]


# [sobel, canny, random, hough, harris, DOG, Laplace, Sato, HOG, gabor, color_hist, fast, Hessian, LBP,
#            Haralick, Euler, Moments]



def load_paintings_from(data_folder, classes, images_in_class, type=".jpg"):
    print("Accessing to database...")
    data_paintings = []
    data_target = []
    for i in range(1, classes + 1):
        for j in range(1, images_in_class + 1):
            image = cv2.imread(f"{data_folder}{i}/{j}{type}")
            if image is not None:
                data_paintings.append(image)
                data_target.append(i - 1)
            else:
                print(f"Error reading image s{i}/{j}{type}")
    print(
        f"Database is uploaded: {len(data_paintings)} paintings, {classes} classes, {images_in_class} images in each class")
    print("=" * 50)
    return [data_paintings, data_target]


def split_data_not_random(data, images_per_class=10, images_per_class_in_train=5):
    amount_of_images = len(data[0])

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        x_train.extend(data[0][i: i + images_per_class_in_train])
        y_train.extend(data[1][i: i + images_per_class_in_train])

        x_test.extend(data[0][i + images_per_class_in_train: i + images_per_class])
        y_test.extend(data[1][i + images_per_class_in_train: i + images_per_class])

    return x_train, x_test, y_train, y_test


def split_data_random(data, images_per_class=10, images_per_person_in_train=5, seed=None, SHOW=False):
    amount_of_images = len(data[0])
    if seed:
        rnd.seed(seed)
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        indexes = list(range(i, i + images_per_class))
        train_indexes = rnd.sample(indexes, images_per_person_in_train)
        # if SHOW:
        #     print(f"train_indexes: {[(ind + 1) % 17 for ind in train_indexes]}")
        x_train.extend([data[0][index] for index in train_indexes])
        y_train.extend([data[1][index] for index in train_indexes])
        train_indexes = sorted(train_indexes)
        to_print = list(set(indexes) - set(train_indexes))
        to_print = [f"{ind + 1}.jpg" if ind <= images_per_class else f"{ind - i + 1}.jpg" for ind in to_print]
        if SHOW: 
            print(f"Test images in class s{i // images_per_class + 1}: {sorted(to_print, key=lambda x: int(x[:-4]))}")
        test_indexes = set(indexes) - set(train_indexes)
        x_test.extend([data[0][index] for index in test_indexes])
        y_test.extend([data[1][index] for index in test_indexes])

    return x_train, x_test, y_train, y_test


def split_data_cross(data, images_per_class=10, images_per_person_in_train=5, train_indxs=[]):
    amount_of_images = len(data[0])

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        indexes = list(range(i, i + images_per_class))
        train_indexes = [indexes[i] for i in train_indxs]
        x_train.extend([data[0][index] for index in train_indexes])
        y_train.extend([data[1][index] for index in train_indexes])

        test_indexes = set(indexes) - set(train_indexes)
        x_test.extend([data[0][index] for index in test_indexes])
        y_test.extend([data[1][index] for index in test_indexes])

    return x_train, x_test, y_train, y_test


def create_feature(images, method):
    return [method(image)[0] for image in images]


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))


def classifier(train, test, method, use_database=None):
    if use_database:
        featured_train = use_database[method.__name__]
    else:
        featured_train = create_feature(train[0], method)
    featured_test = create_feature(test[0], method)
    answers = []
    for test_element in featured_test:
        min_el = [100000, -1]
        for i in range(len(featured_train)):
            dist = distance(test_element, featured_train[i])
            if dist < min_el[0]:
                min_el = [dist, i]
        if min_el[1] < 0:
            answers.append(0)
        else:
            answers.append(train[1][min_el[1]])
    return answers


def voting(train, test, SHOW=False, use_database=None, calc_acc=False, clf=None, cols=3):
    methods = get_methods()
    res = {}
    index = 1
    start = time.time()
    for method in methods:
        res[method.__name__] = classifier(train, test, method, use_database=use_database)
        # print(f"method {method.__name__} ended in {int(time.time() - start)} seconds")
    if clf:
        res["Decision Tree"] = DT_test(test[0], clf)
    voted_answers = []
    _, _, classes = get_size_and_seed()
    for i in range(len(test[0])):
        answers_to_image_1 = {}

        if SHOW:
            plt.subplot(3, cols, index)
            index += 1
            plt.imshow(cv2.cvtColor(test[0][i], cv2.COLOR_BGR2RGB)), plt.axis("off"), plt.title("Query Image")

        for method in res:
            answer = res[method][i]
            if answer in answers_to_image_1:
                answers_to_image_1[answer] += 1
            else:
                answers_to_image_1[answer] = 1
            if method == "color_hist" and answers_to_image_1[answer]:
                answers_to_image_1[answer] += 0.5
            if method == "LBP" and answers_to_image_1[answer]:
                answers_to_image_1[answer] += 0.5
            if method == "Decision Tree" and answers_to_image_1[answer]:
                answers_to_image_1[answer] += 0.5

            if SHOW:
                plt.subplot(3, cols, index)
                # for train_image, true_answer in zip(train[0], train[1]):
                #     if true_answer == answer:
                        # plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.axis("off"), plt.title(method)
                        # break
                plt.axis("off"), plt.title(method)
                plt.text(0.3,
                        0.5,
                        classes[answer],
                        transform=plt.gca().transAxes, fontdict={'size': 12})
                index += 1
                        
        best_size = sorted(answers_to_image_1.items(), key=lambda item: item[1], reverse=True)[0]
        voted_answers.append(best_size[0])

    if calc_acc:
        sum = 0
        for i in range(len(test[0])):
            if test[1][i] == voted_answers[i]:
                sum += 1
        print(f"Accuracy: {sum / len(test[0])}")

    return voted_answers


def test_voting(train, test, use_database=None, clf=None):
    res = voting(train, test, use_database=use_database, clf=clf)
    sum = 0
    for i in range(len(test[0])):
        if test[1][i] == res[i]:
            sum += 1
    return sum / len(test[0])


def draw_methods(image):
    plt.rcParams["figure.figsize"] = (20, 20)
    index = 1
    cols = len(get_methods()) // 2 + 1
    plt.subplot(2, cols, index, title="original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    for method in get_methods():
        index += 1
        plt.subplot(2, cols, index, title=method.__name__)
        to_draw = method(image)[1]
        if method == color_hist:
            for hist, col in zip(to_draw, ('b', 'g', 'r')):
                plt.plot(range(len(hist)), hist, col)
        elif method in (LBP, ):
            plt.plot(range(len(to_draw)), to_draw)
        elif method == Haralick:
            plt.imshow(to_draw, cmap="gray")
        elif method in (Euler,):
            plt.text(0.35,
                     0.5,
                     to_draw,
                     transform=plt.gca().transAxes, fontdict={'size': 20}), plt.axis("off")
        elif len(to_draw.shape) == 2:
            plt.imshow(to_draw, cmap="gray"), plt.axis("off")
        else:
            plt.imshow(cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)), plt.axis("off")
    plt.show()


def test_methods(train, test, clf=None):
    res = {}
    for method in get_methods():
        answers = classifier(train, test, method)
        correct_answers = 0
        for i in range(len(answers)):
            if answers[i] == test[1][i]:
                correct_answers += 1
        res[method.__name__] = correct_answers / len(answers)
    answers = DT_test(test[0], clf=clf)
    correct_answers = 0
    for i in range(len(answers)):
        if answers[i] == test[1][i]:
            correct_answers += 1
    res["Decision Tree"] = correct_answers / len(answers)
    return res


def get_size_and_seed():
    size = 10
    seed = 10
    # 10 PAINTS IN TRAIN, 6 PAINTS IN TEST, seed=10
    # methods:  {'color_hist': 0.7083333333333334, 'fast': 0.5, 'random': 0.5, 'LBP': 0.3333333333333333, 'Decision Tree': 0.7083333333333334}
    # voting --> 0.8333333333333334
    classes = {0: "Шишкин", 1: "Айвазовский",
               2: "Пикассо", 3: "Суриков"}
    return size, seed, classes


if __name__ == "__main__":
    size, seed, classes = get_size_and_seed()
    data = load_paintings_from("./Paints/s", len(classes), 16)
    # for image in data[0]:
    #     draw_methods(image)

    stats = {}
    for seed in range(1, 30):
        print(f"seed={seed}")
        # for size in range(1, 16):
        x_train, x_test, y_train, y_test = split_data_random(data, 16, size, seed=seed)
        train = [x_train, y_train]
        test = [x_test, y_test]

        methods = get_methods()
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=0)
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
        clf = clf.fit(tree_train, y_train)

        res = test_methods(train, test, clf=clf)
        classf = test_voting(train, test, clf=clf)
        # for method in get_methods():
        #     if method.__name__ in stats:
        #         stats[method.__name__].append(res[method.__name__])
        #     else:
        #         stats[method.__name__] = [res[method.__name__]]
        # if "voting" in stats:
        #     stats["voting"].append(classf)
        # else:
        #     stats["voting"] = [classf]
        if classf >= 0.7:
            print(f"{size} PAINTS IN TRAIN, {16 - size} PAINTS IN TEST, seed={seed}")
            print("methods: ", res)
            print(f"voting --> {classf}")
            print("*" * 10)
    # for method, stat in stats.items():
    #     plt.plot(range(len(stat)), stat, label=method)
    # plt.title(f"train size={size}"), plt.legend(loc='best'), plt.xlabel("partition"), plt.ylabel("score")
    # plt.show()

    # count = 0
    # summ = 0
    # result = []
    # for test_image, true_answer in zip(x_test, y_test):
    #
    #     res = voting(train, [[test_image], [true_answer]])
    #     # res = classifier(train, test, color_hist)
    #     if true_answer == res[0]:
    #         summ += 1
    #     else:
    #         print(f"return {classes[res[0]]} but true is {classes[true_answer]}")
    #         # res = {}
    #         # for method in get_methods():
    #         #     answers = classifier(train, [[test_image], [true_answer]], method)
    #         #     print(f"method {method.__name__} found {classes[answers[0]]}")
    #         # print(test_methods(train, [[test_image], [true_answer]]))
    #         # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    #         # plt.show()
    #     count += 1
    #     result.append(summ / count)
    #     print(f"{count} images --> {summ / count}")
    # plt.plot(range(1, len(result) + 1),  result), plt.xlabel("amount of test images"), plt.ylabel("score"), plt.title("Voting")
    # plt.show()

    # count = 1
    # plt.rcParams["figure.figsize"] = (10, 6)
    # index = 1
    # for train_image, ans in zip(x_train, y_train):
    #     plt.subplot(2, 5, index)
    #     plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    #     index += 1
    #     if index > 10:
    #         plt.savefig(f"./Results/{count}.jpg")
    #         count += 1
    #         index = 1
