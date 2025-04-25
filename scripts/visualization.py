import plotly.express as px

class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def get_data_overview(self):
        return self.df.head(20), self.df.describe()

    def get_risk_distribution(self):
        fig_risk = px.histogram(self.df, x="Risk", nbins=20, color_discrete_sequence=['#4F8BF9'])
        return fig_risk

    def get_correlation(self):
        corr = self.df.corr(numeric_only=True)
        fig_corr = px.imshow(corr, color_continuous_scale='Blues', aspect="auto")
        return fig_corr

    def get_interactive_scatter(self, x_axis, y_axis):
        fig_scatter = px.scatter(self.df, x=x_axis, y=y_axis, color="Risk", color_continuous_scale='Bluered',
                                title=f"{y_axis} en fonction de {x_axis}")
        return fig_scatter 