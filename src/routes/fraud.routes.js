import express from 'express';
import { getFraudAnalytics, checkTransactionFraud } from '../controller/fraud.controller.js';
import { isAuthenticated } from '../middleware/authMiddleware.js';

const router = express.Router();

// Get fraud analytics for admin dashboard
router.get('/analytics', isAuthenticated, getFraudAnalytics);

// Check single transaction for fraud
router.post('/check', isAuthenticated, checkTransactionFraud);

export default router;
