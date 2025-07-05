import express from 'express';
import {
  signup,
  login,
  logout,
  profile,
} from '../controller/user.controller.js';
import { isAuthenticated } from '../middleware/authMiddleware.js';

const router = express.Router();

// Public routes
router.post('/signup', signup);
router.post('/login', login);
router.post('/logout', logout);
router.get('/profile', isAuthenticated, profile);


export default router;
