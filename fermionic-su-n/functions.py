import numpy as np

#------------------------------------------------------------------------------
#------------------------- Methods for linear algebra -------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#------------------------- Methods for combinatorics --------------------------
#------------------------------------------------------------------------------

def reduced_matrix(m, row_indices, column_indices):
    return(np.delete(np.delete(m, row_indices, axis = 0), column_indices, axis = 1))

def eta(i, l, offset = 0):
    # number of elements in l smaller than i
    if isinstance(i, int) or isinstance(i, np.int64):
        res = 0
        for obj in l:
            if obj < i + offset:
                res += 1
        return(res)
    else:
        res = 0
        for obj in i:
            res += eta(obj, l, offset)
        return(res)

def sign(i):
    if i % 2 == 0:
        return(1.0)
    else:
        return(-1.0)

def subset_indices(N, k):
    # returns a list of all subsequences of <N> of length k
    if k == 0:
        return([[]])
    if len(N) == 0:
        return([[]])
    res = []
    for k_1 in range(0, len(N) - k + 1):
        minor = subset_indices(N[k_1+1:], k-1)
        for m in minor:
            res.append([N[k_1]] + m)
    return(res)

# Function for permutation signature (parity)
def permutation_signature(arr):
    # arr is a list of distinct values
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, inv_left = merge_sort(arr[:mid])
        right, inv_right = merge_sort(arr[mid:])
        merged, inv_split = merge_and_count(left, right)
        return merged, inv_left + inv_right + inv_split

    def merge_and_count(left, right):
        merged = []
        i = j = inv_count = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i
                j += 1
        merged += left[i:]
        merged += right[j:]
        return merged, inv_count

    _, inversions = merge_sort(arr)
    return 1 if inversions % 2 == 0 else -1

def choose_iterator(choice):
    # For a choice of S elements out of M boxes, this function returns the next
    # choice or None if the provided choice was the last choice.
    M = len(choice)
    S = sum(choice)
    number_of_passed_zeros = 0
    pointer_index = 0

    while(choice[pointer_index] == 0):
        number_of_passed_zeros += 1
        pointer_index += 1

        if number_of_passed_zeros == M - S:
            # We reached the end
            return(None)
    while(choice[pointer_index] == 1):
        pointer_index += 1
    return([1] * (pointer_index - 1 - number_of_passed_zeros) + [0] * (number_of_passed_zeros + 1) + [1] + choice[pointer_index+1:])

# -----------------------------------------------------------------------------
# ------------------------ Methods for random sampling ------------------------
# -----------------------------------------------------------------------------

# ---------- Sampling a vector using a self-correlation matrix: \Sigma_
def sample_with_autocorrelation(mu, Sigma, n_samples):
    # mu is the mean value vector
    # Sigma is the auto-correlation matrix
    L = np.linalg.cholesky(Sigma)  # or use eigh for numerical safety
    Z = np.random.randn(n_samples, Sigma.shape[0])
    return(Z @ L.T + mu) # result[n_i][i] = X_i in the n_i-th sample

def sample_with_autocorrelation_safe(Sigma, n_samples):
    # same as above, but enforces positive-semidefiniteness by clipping eigvals
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.clip(eigvals, 0, None)
    L = eigvecs * np.sqrt(eigvals)[None, :]
    Z = np.random.randn(n_samples, Sigma.shape[0])
    return Z @ L.T  # shape (n_samples, n_x)


# -----------------------------------------------------------------------------
# ------------------------ Methods for data formatting ------------------------
# -----------------------------------------------------------------------------

def dtstr(seconds, max_depth = 2):
    # Dynamically chooses the right format
    # max_depth is the number of different measurements (e.g. max_depth = 2: "2 days 5 hours")
    if seconds >= 60 * 60 * 24:
        # Days
        if max_depth == 1:
            return(f"{int(np.round(seconds / (60 * 60 * 24)))} days")
        remainder = seconds % (60 * 60 * 24)
        days = int((seconds - remainder) / (60 * 60 * 24))
        return(f"{days} days {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60 * 60:
        # Hours
        if max_depth == 1:
            return(f"{int(np.round(seconds / (60 * 60)))} hours")
        remainder = seconds % (60 * 60)
        hours = int((seconds - remainder) / (60 * 60))
        return(f"{hours} hours {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60:
        # Minutes
        if max_depth == 1:
            return(f"{int(np.round(seconds / 60))} min")
        remainder = seconds % (60)
        minutes = int((seconds - remainder) / (60))
        return(f"{minutes} min {dtstr(remainder, max_depth - 1)}")
    if seconds >= 1:
        # Seconds
        if max_depth == 1:
            return(f"{int(np.round(seconds))} sec")
        remainder = seconds % (1)
        secs = int((seconds - remainder))
        return(f"{secs} sec {dtstr(remainder, max_depth - 1)}")
    # Milliseconds
    return(f"{int(np.round(seconds / 0.001))} ms")

# ------------------------------ Plot formatting ------------------------------

def subplot_dimensions(number_of_plots):
    # 1x1, 2x1, 3x1, 2x2, 3x2, 4x2, 3x3, 4x3, 5x3
    if number_of_plots == 1:
        return(1, 1)
    if number_of_plots == 2:
        return(2, 1)
    if number_of_plots == 3:
        return(3, 1)
    if number_of_plots == 4:
        return(2, 2)
    if number_of_plots <= 6:
        return(3, 2)
    if number_of_plots <= 8:
        return(4, 2)
    if number_of_plots <= 9:
        return(3, 3)
    if number_of_plots <= 12:
        return(4, 3)
    if number_of_plots <= 15:
        return(5, 3)
    return(int(np.ceil(np.sqrt(number_of_plots))), int(np.ceil(np.sqrt(number_of_plots))))

def m_ext(m):
    # extent of matrix for imshow heatmaps
    return((0, m.shape[1], m.shape[0], 0))


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts





# Some other helpful properties

ref_energy_colors = {
    "ref state" : "#b3b3ff",
    "full CI" : "#005ce6",
    "LE CI" : "#80ffff",
    "trimmed CI" : "#6699ff"
    }

