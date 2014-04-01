

#ifndef __ZSTRING_H__
#define __ZSTRING_H__

#include <z.h>
#include <vector>
#include <string>
#include <ostream>
#include <iostream>
#include <sstream>
#include <cstring>

using std::string;
using std::vector;
using std::stringstream;
using std::exception;

template <typename T> static inline string zString(const T &x);

static inline void zString_replace(string &value, string const &search,
                                    string const &replace) {
  for (string::size_type next = value.find(search); next != string::npos;
       next = value.find(search, next)) {
    value.replace(next, search.length(), replace);
    next += replace.length();
  }
}

static inline string zString_quote(string str) {
  string s = str;
  zString_replace(s, "\\", "\\\\");
  s = "\"" + s + "\"";
  return s;
}

static inline string zString_quote(const char *str) {
  if (str == NULL) {
    return zString_quote("");
  } else {
    return zString_quote(string(str));
  }
}

static inline char *zString_duplicate(const char *str) {
  if (str == NULL) {
    return NULL;
  } else {
    char *newstr;
    size_t len = strlen(str);
    newstr = zNewArray(char, len + 1);
    memcpy(newstr, str, len * sizeof(char));
    newstr[len] = '\0';
    return newstr;
  }
}

static inline char *zString_duplicate(std::string str) {
  return zString_duplicate(str.c_str());
}

static inline string zString(void) {
  string s = "";
  return s;
}

template <typename T> static inline string zString(const T &x) {
  try {
    stringstream ss;
    ss << x;
    return ss.str();
  }
  catch (exception & e) {
    return string();
  }
}

template <> inline string zString(const bool &x) {
  return x ? "True" : "False";
}

template <> inline string zString(const vector<string> &x) {
  stringstream ss;
  ss << "{";
  for (size_t ii = 0; ii < x.size(); ii++) {
    ss << zString_quote(x[ii]);
    if (ii != x.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}";

  return ss.str();
}

template <> inline string zString(const vector<int> &x) {
  stringstream ss;
  ss << "{";
  for (size_t ii = 0; ii < x.size(); ii++) {
    ss << x[ii];
    if (ii != x.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}";

  return ss.str();
}

template <> inline string zString(const vector<double> &x) {
  stringstream ss;
  ss << "{";
  for (size_t ii = 0; ii < x.size(); ii++) {
    ss << x[ii];
    if (ii != x.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}";

  return ss.str();
}

template <typename T0, typename T1>
static inline string zString(const T0 &x0, const T1 &x1) {
  stringstream ss;
  ss << zString(x0) << zString(x1);

  return ss.str();
}

template <typename T0, typename T1, typename T2>
static inline string zString(const T0 &x0, const T1 &x1, const T2 &x2) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2);
  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5>
static inline string zString(const T0 &x0, const T1 &x1, const T2 &x2,
                              const T3 &x3, const T4 &x4, const T5 &x5) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7, const T8 &x8) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7)
     << zString(x8);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7, const T8 &x8, const T9 &x9) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7)
     << zString(x8) << zString(x9);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9,
          typename T10>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7, const T8 &x8, const T9 &x9,
         const T10 &x10) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7)
     << zString(x8) << zString(x9) << zString(x10);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9,
          typename T10, typename T11>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7, const T8 &x8, const T9 &x9,
         const T10 &x10, const T11 &x11) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7)
     << zString(x8) << zString(x9) << zString(x10) << zString(x11);

  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9,
          typename T10, typename T11, typename T12>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7, const T8 &x8, const T9 &x9,
         const T10 &x10, const T11 &x11, const T12 &x12) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7)
     << zString(x8) << zString(x9) << zString(x10) << zString(x11)
     << zString(x12);
  return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9,
          typename T10, typename T11, typename T12, typename T13>
static inline string
zString(const T0 &x0, const T1 &x1, const T2 &x2, const T3 &x3, const T4 &x4,
         const T5 &x5, const T6 &x6, const T7 &x7, const T8 &x8, const T9 &x9,
         const T10 &x10, const T11 &x11, const T12 &x12, const T13 &x13) {
  stringstream ss;
  ss << zString(x0) << zString(x1) << zString(x2) << zString(x3)
     << zString(x4) << zString(x5) << zString(x6) << zString(x7)
     << zString(x8) << zString(x9) << zString(x10) << zString(x11)
     << zString(x12) << zString(x13);
  return ss.str();
}

template <typename X, typename Y>
static inline zBool zString_sameQ(const X &x, const Y &y) {
  string xs = zString(x);
  string ys = zString(y);
  return strcmp(xs.c_str(), ys.c_str()) == 0;
}

static inline zBool zString_sameQ(const string &x, const string &y) {
  return x.compare(y) == 0;
}

static inline char *zString_toLower(const char *str) {
  if (str == NULL) {
    return NULL;
  } else {
    char *res, *iter;

    res = iter = zString_duplicate(str);
    while (*iter != '\0') {
      *iter++ = tolower(*str++);
    }
    return res;
  }
}

static inline zBool zString_startsWith(const char *str, const char *prefix) {
  while (*prefix != '\0') {
    if (*str == '\0' || *str != *prefix) {
      return zFalse;
    }
    str++;
    prefix++;
  }
  return zTrue;
}

#endif /* __ZSTRING_H__ */

