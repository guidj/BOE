import base64
import io


def generate_metric_figure_from_matrix(steps, matrix, arms, ax, title, ci_matrix=None):
    ax.set_title(title)
    for idx, arm in enumerate(arms):
        values = matrix[:, idx]
        line, = ax.plot(steps, values)
        if len(arms) < 10:
            ax.legend()
        if ci_matrix is not None:
            ci = ci_matrix[:, idx]
            upper_ci = values + ci
            lower_ci = values - ci
            ax.fill_between(steps, lower_ci, upper_ci, alpha=0.4, label='CI')
        ax.text(steps[-1], values[-1], str(arm.name))
        ax.ticklabel_format(style='sci')
        line.set_label(idx)


def generate_metric_figure_from_values(steps, values, title, ax, label, ci=None):
    ax.set_title(title)
    ax.plot(steps, values, label=label)
    if ci is not None:
        upper_ci = values + ci
        lower_ci = values - ci
        ax.fill_between(steps, lower_ci, upper_ci, alpha=0.4, label='CI')

    ax.legend()

    return ax


def encode_figure_as_base64(figure):
    img = io.BytesIO()
    figure.savefig(img, format='png',
                   bbox_inches='tight')
    img.seek(0)

    data = base64.b64encode(img.getvalue())
    return data
