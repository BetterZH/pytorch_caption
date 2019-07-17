import matplotlib.pyplot as plt
import skimage

def show_caption_att(img, caption, probs, weights):

    print "Sampled Caption: %s" % caption

    # Plot original image
    plt.subplot(4, 5, 1)
    plt.imshow(img)
    plt.axis('off')

    # Plot images with attention weights
    words = caption.split(" ")
    for t in range(len(words)):
        word = words[t]
        plt.subplot(4, 5, t + 2)
        plt.text(0, 1, '%s(%.2f)' % (word, probs[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(img)
        alp_curr = weights[t, :].reshape(14, 14)
        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
        plt.imshow(alp_img, alpha=0.85)
        plt.axis('off')
    plt.show()