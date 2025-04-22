# Install necessary libraries if you haven't already
# install.packages(c("ggplot2", "plotly", "lattice", "highcharter", "leaflet"))

# Load libraries
library(ggplot2)
library(plotly)
library(lattice)
library(highcharter)
library(leaflet)

# Prepare data
data(mtcars)
df <- mtcars
df$cyl <- as.factor(df$cyl)
df$car <- rownames(df)

# 1. ggplot2 - Scatter Plot
ggplot(df, aes(x = wt, y = mpg, color = cyl)) +
  geom_point(size = 3) +
  labs(title = "MPG vs Weight", x = "Weight", y = "MPG") +
  theme_minimal()

# 2. plotly - Interactive Bar Chart
cyl_counts <- as.data.frame(table(df$cyl))
names(cyl_counts) <- c("Cylinders", "Count")
p <- ggplot(cyl_counts, aes(x = Cylinders, y = Count, fill = Cylinders)) +
  geom_bar(stat = "identity") +
  labs(title = "Number of Cars per Cylinder") +
  theme_minimal()
ggplotly(p)  # Convert to interactive plot

# 3. lattice - Box Plot
bwplot(mpg ~ cyl, data = df, main = "MPG by Cylinder (Lattice)",
       xlab = "Cylinders", ylab = "MPG")

# 4. highcharter - Line Plot
highchart() %>%
  hc_title(text = "Line Plot of MPG for Cars") %>%
  hc_xAxis(categories = df$car) %>%
  hc_add_series(name = "MPG", data = df$mpg, type = "line")

# 5. leaflet - Interactive Map
leaflet() %>%
  addTiles() %>%
  addMarkers(lng = 72.8777, lat = 19.0760, popup = "📍 Mumbai")

