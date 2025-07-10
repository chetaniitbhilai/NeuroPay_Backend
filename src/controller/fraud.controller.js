import Payment from '../models/payment.model.js';
import axios from 'axios';

// Function to call fraud detection service
async function checkFraud(paymentData) {
  try {
    const response = await axios.post('http://localhost:5001/predict', paymentData);
    return response.data;
  } catch (error) {
    console.error('Error during fraud detection:', error);
    return null;
  }
}

// Get fraud analytics for admin dashboard
export const getFraudAnalytics = async (req, res) => {
  try {
    // Get overall fraud statistics
    const totalPayments = await Payment.countDocuments();
    const fraudPayments = await Payment.countDocuments({ 'fraudDetection.isFraud': true });
    const checkedPayments = await Payment.countDocuments({ 'fraudDetection.isChecked': true });

    // Get risk level distribution
    const riskDistribution = await Payment.aggregate([
      { $match: { 'fraudDetection.isChecked': true } },
      { $group: { _id: '$fraudDetection.riskLevel', count: { $sum: 1 } } }
    ]);

    // Get fraud trends over time (last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const fraudTrends = await Payment.aggregate([
      {
        $match: {
          'fraudDetection.isChecked': true,
          date: { $gte: thirtyDaysAgo }
        }
      },
      {
        $group: {
          _id: {
            year: { $year: '$date' },
            month: { $month: '$date' },
            day: { $dayOfMonth: '$date' }
          },
          totalTransactions: { $sum: 1 },
          fraudTransactions: {
            $sum: { $cond: ['$fraudDetection.isFraud', 1, 0] }
          },
          avgFraudProbability: { $avg: '$fraudDetection.fraudProbability' }
        }
      },
      { $sort: { '_id.year': 1, '_id.month': 1, '_id.day': 1 } }
    ]);

    // Get top fraudulent payment methods
    const fraudByMethod = await Payment.aggregate([
      { $match: { 'fraudDetection.isFraud': true } },
      { $group: { _id: '$method', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ]);

    // Get recent high-risk transactions
    const highRiskTransactions = await Payment.find({
      'fraudDetection.riskLevel': 'HIGH'
    })
      .sort({ date: -1 })
      .limit(10)
      .populate('userId', 'name email');

    res.status(200).json({
      overview: {
        totalPayments,
        fraudPayments,
        checkedPayments,
        fraudRate: totalPayments > 0 ? (fraudPayments / totalPayments * 100).toFixed(2) : 0
      },
      riskDistribution,
      fraudTrends,
      fraudByMethod,
      highRiskTransactions
    });
  } catch (error) {
    console.error('Error fetching fraud analytics:', error);
    res.status(500).json({ error: 'Failed to fetch fraud analytics' });
  }
};

// Check single transaction for fraud
export const checkTransactionFraud = async (req, res) => {
  try {
    const { paymentId } = req.body;

    if (!paymentId) {
      return res.status(400).json({ error: 'Payment ID is required' });
    }

    // Find the payment
    const payment = await Payment.findById(paymentId);

    if (!payment) {
      return res.status(404).json({ error: 'Payment not found' });
    }

    // Check for fraud
    const now = new Date().toISOString();
    const fraudItems = payment.items.map(item => ({
      Time: now,
      Source: payment.userId,
      Target: payment.merchantId,
      Amount: item.price * item.quantity,
      Location: payment.location,
      Type: payment.method,
    }));
    const fraudResult = await checkFraud({ items: fraudItems });

    if (fraudResult) {
      // Update payment with fraud detection results
      payment.fraudDetection = {
        isChecked: true,
        fraudProbability: fraudResult.fraud_probability,
        prediction: fraudResult.prediction,
        riskLevel: fraudResult.risk_level,
        isFraud: fraudResult.is_fraud,
        modelUsed: fraudResult.model_used,
        checkedAt: new Date()
      };

      await payment.save();

      res.status(200).json({
        message: 'Fraud check completed',
        fraudResult,
        payment
      });
    } else {
      res.status(500).json({ error: 'Fraud detection service unavailable' });
    }
  } catch (error) {
    console.error('Error checking transaction fraud:', error);
    res.status(500).json({ error: 'Failed to check transaction fraud' });
  }
};
