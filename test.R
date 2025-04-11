install.packages("QR")
library(QR)

M.j <- matrix(c(0, 0, 0, 0, 0, 0, 0,   # Row 1
                0, 0, 0, 0, 0, 0, 0,   # Row 2
                0, 0, 0.7871478, 0.1978098, 0.4109216, -0.383821, -0.1584128),  # Row 3
              nrow = 3, ncol = 7, byrow = TRUE)

# Use the QR package function
qr_result <- QR(t(M.j), complete=T)
# Extract the Q matrix
qr_result$Q
