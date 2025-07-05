import mongoose from 'mongoose';

const paymentSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: 'User',
  },
  amount: {
    type: Number,
    required: true,
  },
  items: [
    {
      _id: String,
      name: String,
      price: Number,
      quantity: Number,
      total: Number,
      image: String,
    },
  ],
  paymentIntentId: {
    type: String,
    required: true,
  },
  status: {
    type: String,
    required: true,
  },
  date: {
    type: Date,
    default: Date.now, // ✅ This sets default to current date/time
  },
});

const Payment = mongoose.model('Payment', paymentSchema);

export default Payment;
