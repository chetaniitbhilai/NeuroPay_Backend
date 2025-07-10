import express from "express";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import cors from "cors";
import connectToMongoDB from "./src/db/connectToMongoDB.js";
import productRoutes from "./src/routes/productRoute.js";
import userRoutes from "./src/routes/userRoute.js";
import paymentRoutes from "./src/routes/payment.routes.js";
import fraudRoutes from "./src/routes/fraud.routes.js";

dotenv.config();

console.log("MONGO_DB_URI:", process.env.MONGO_DB_URI);
console.log("PORT:", process.env.PORT);

const PORT = process.env.PORT || 5002;
const app = express();

app.use(express.static('public'));
app.use(cors({
    origin: '*',
    credentials: true,
}));
app.use(express.json());
app.use(cookieParser());

app.use("/api", productRoutes);
app.use("/api/auth", userRoutes);
app.use('/api/payments', paymentRoutes);
app.use('/api/fraud', fraudRoutes);

const startServer = async () => {
    try {
        await connectToMongoDB();
        app.listen(PORT, () => {
            console.log(`🚀 Server listening on port ${PORT}`);
        });
    } catch (err) {
        console.error("❌ Failed to start server:", err.message);
        process.exit(1);
    }
};

startServer();