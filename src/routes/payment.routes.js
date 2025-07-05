import express from 'express';
import { createPaymentIntent } from '../controller/payment.controller.js';
import { isAuthenticated } from '../middleware/authMiddleware.js';

const router = express.Router();

router.post('/create-intent', isAuthenticated, createPaymentIntent); // 👈 Expects amount + userId

export default router;
