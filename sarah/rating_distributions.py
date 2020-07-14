import matplotlib.pyplot as plt


def rating_distribution(filename):
    ratings = [0] * 5
    with open(filename, 'r') as file:
        file.readline()  # remove the header, don't save it, because we don't need it
        for line in file:
            entry, prediction = line.split(',')  # split the movie rating from the ID
            star_rating = int(prediction)  # save the rating
            ratings[star_rating-1] += 1
    return ratings


def plot_ratings(rating_array):
    x = [1, 2, 3, 4, 5]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.bar(x, rating_array)
    plt.savefig('rating_dist.png')


star_ratings = rating_distribution('../input/data_train.csv')
plot_ratings(star_ratings)


