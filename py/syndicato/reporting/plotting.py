import base64
import io

import matplotlib.pyplot as plt


def generate_metric_figure_from_matrix(matrix, arms, ax, title, ci_matrix=None):
    ax.set_title(title)

    for idx, arm in enumerate(arms):
        values = matrix[idx, :]
        line, = ax.plot(values)
        if ci_matrix is not None:
            ci = ci_matrix[idx, :]
            upper_ci = values + ci
            lower_ci = values - ci
            ax.fill_between(range(values.size), lower_ci, upper_ci, alpha=0.4, label='CI')
        ax.text(values.size, values[-1], str(arm.name))
        line.set_label(idx)

        if len(arms) < 10:
            plt.legend()
        else:
            pass
            # TODO: optimizations to the plot with many lines


def generate_metric_figure_from_values(values, title, ax, label, ci=None):
    ax.set_title(title)
    ax.plot(values, label=label)
    if ci is not None:
        upper_ci = values + ci
        lower_ci = values - ci
        ax.fill_between(range(values.size), lower_ci, upper_ci, alpha=0.4, label='CI')

    plt.legend()

    return ax


def encode_figure_as_base64(figure):
    img = io.BytesIO()
    figure.savefig(img, format='png',
                   bbox_inches='tight')
    img.seek(0)

    data = base64.b64encode(img.getvalue())
    return data
