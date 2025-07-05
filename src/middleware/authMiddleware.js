import jwt from 'jsonwebtoken';

export const isAuthenticated = (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const token = authHeader.split(' ')[1];

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    console.log("Decoded:", decoded); // should show { userID: '...' }

    // ✅ Add custom `user` field to request
    req.user = { id: decoded.userID }; 
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};
