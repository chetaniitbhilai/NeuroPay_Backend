import express from 'express';
import { createPaymentIntent, getUserSuccessfulPayments, logUPIPayment, markPaymentSuccess } from '../controller/payment.controller.js';
import { isAuthenticated } from '../middleware/authMiddleware.js';

const router = express.Router();

router.post('/create-intent', isAuthenticated, createPaymentIntent); // 👈 Expects amount + userId
router.post('/upi', isAuthenticated, logUPIPayment);
router.post('/mark-success', isAuthenticated, markPaymentSuccess);
router.get('/history', isAuthenticated, getUserSuccessfulPayments);


export default router;
